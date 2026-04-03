"""Helm chart manifest policy tests.

Renders the chart with both the default values.yaml and the local k3d overlay,
then asserts structural invariants that CI would never catch by running unit
tests alone.

These tests catch configuration bugs such as:
  - Missing required env vars in values.yaml (pod crashes at startup)
  - Secret key references that point to non-existent keys in the Secret
  - Services whose selectors don't match any Deployment's pod labels
  - Deployments missing readiness probes (pod receives traffic before ready)
  - Hardcoded credentials in plain env.value fields

Requirements: helm binary must be on PATH.
Run from repo root: PYTHONPATH=. pytest tests/helm/ -v
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

# Skip the entire module when helm is not installed.
# In CI, azure/setup-helm@v4 installs it before running this file.
pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm binary not found on PATH — install helm 3 to run chart tests",
)

CHART_DIR = Path("helm/ml-system")
VALUES_DEFAULT = CHART_DIR / "values.yaml"
VALUES_LOCAL = CHART_DIR / "values-local.yaml"

# All vars that serving/main.py and ModelStore call require_env() on at startup.
# A Deployment rendered without these will crash immediately on first request.
_SERVING_REQUIRED_ENV_VARS = {
    "MODEL_NAME",
    "MODEL_STAGE",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_S3_ENDPOINT_URL",
    "SERVING_SIMULATED_LATENCY_MS",
}

# Credential substrings that must never appear as plain env.value in the serving pod.
# These values should only flow through Kubernetes Secrets via secretKeyRef.
_FORBIDDEN_PLAINTEXT_IN_SERVING = [
    "minioadmin",
    "AKIAIOSFODNN7EXAMPLE",
    "wJalrXUtnFEMI",
]


# ── helpers ───────────────────────────────────────────────────────────────────


def _helm_lint(values_files: list[Path]) -> None:
    cmd = ["helm", "lint", str(CHART_DIR)]
    for vf in values_files:
        cmd += ["-f", str(vf)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"helm lint failed:\n{result.stdout}\n{result.stderr}"


def _render_chart(values_files: list[Path]) -> list[dict]:
    """Return all non-None YAML documents produced by helm template."""
    cmd = ["helm", "template", "ml-system", str(CHART_DIR)]
    for vf in values_files:
        cmd += ["-f", str(vf)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"helm template failed:\n{result.stdout}\n{result.stderr}"
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc is not None]


def _env_names(containers: list[dict]) -> set[str]:
    """Return all env var names declared across a list of container specs."""
    names: set[str] = set()
    for c in containers:
        for entry in c.get("env") or []:
            names.add(entry["name"])
    return names


def _iter_containers(docs: list[dict]):
    """Yield (doc, container_spec) for every container and initContainer in the manifests."""
    for doc in docs:
        spec = doc.get("spec", {})
        # Deployment / DaemonSet / StatefulSet have spec.template.spec
        pod_spec = spec.get("template", {}).get("spec", {})
        for c in pod_spec.get("containers") or []:
            yield doc, c
        for c in pod_spec.get("initContainers") or []:
            yield doc, c
        # Bare Pods
        for c in spec.get("containers") or []:
            yield doc, c


# ── helm lint ─────────────────────────────────────────────────────────────────


def test_helm_lint_default_values():
    _helm_lint([VALUES_DEFAULT])


def test_helm_lint_local_values():
    _helm_lint([VALUES_DEFAULT, VALUES_LOCAL])


# ── env var completeness ──────────────────────────────────────────────────────


def test_serving_required_env_vars_present_in_default_values():
    """All require_env() calls in serving/main.py and ModelStore must be satisfied
    by values.yaml alone, without any overlay.

    This directly tests the gap where vars appear only in values-local.yaml, so
    any deployment that omits the overlay causes an immediate pod crash.
    """
    docs = _render_chart([VALUES_DEFAULT])

    serving_deployments = [
        d
        for d in docs
        if d.get("kind") == "Deployment" and d.get("metadata", {}).get("name") == "fastapi-serving"
    ]
    assert serving_deployments, "fastapi-serving Deployment not found in rendered chart"

    containers = serving_deployments[0]["spec"]["template"]["spec"]["containers"]
    present = _env_names(containers)
    missing = _SERVING_REQUIRED_ENV_VARS - present
    assert not missing, (
        f"Serving Deployment (values.yaml only) is missing required env vars: {sorted(missing)}.\n"
        "Add them to helm/ml-system/values.yaml serving.env so deployments without\n"
        "values-local.yaml don't crash at startup."
    )


# ── readiness probes ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "values_files,label",
    [
        ([VALUES_DEFAULT], "default"),
        ([VALUES_DEFAULT, VALUES_LOCAL], "local"),
    ],
    ids=["default", "local"],
)
def test_all_deployments_have_readiness_probe(values_files, label):
    """Without a readinessProbe, Kubernetes routes traffic to a pod before it
    is accepting connections, causing 502s during rollouts.
    """
    docs = _render_chart(values_files)
    for doc in docs:
        if doc.get("kind") != "Deployment":
            continue
        name = doc["metadata"]["name"]
        containers = doc["spec"]["template"]["spec"].get("containers") or []
        for c in containers:
            assert c.get("readinessProbe"), (
                f"Container '{c['name']}' in Deployment '{name}' ({label} values) "
                f"has no readinessProbe. Add one so k8s only routes traffic when ready."
            )


# ── secret key reference integrity ───────────────────────────────────────────


def test_secret_key_refs_resolve_to_declared_keys():
    """Every secretKeyRef that references ml-system-secrets must use a key
    that actually exists in the Secret's stringData.

    Mismatched keys cause pods to stay in CreateContainerConfigError indefinitely.
    """
    docs = _render_chart([VALUES_DEFAULT])

    declared_keys: set[str] = set()
    for doc in docs:
        if doc.get("kind") == "Secret" and doc["metadata"]["name"] == "ml-system-secrets":
            declared_keys = set((doc.get("stringData") or {}).keys())
            break
    assert declared_keys, "ml-system-secrets Secret not found or has no stringData"

    for doc, container in _iter_containers(docs):
        for entry in container.get("env") or []:
            ref = (entry.get("valueFrom") or {}).get("secretKeyRef") or {}
            if ref.get("name") == "ml-system-secrets":
                key = ref["key"]
                assert key in declared_keys, (
                    f"Container '{container['name']}' in '{doc['metadata']['name']}' "
                    f"references secret key '{key}' which is not declared in "
                    f"ml-system-secrets.stringData.\n"
                    f"Declared keys: {sorted(declared_keys)}"
                )


# ── no hardcoded credentials in serving ──────────────────────────────────────


def test_no_hardcoded_credentials_in_serving_env():
    """Known credential strings must not appear as plain env.value in the serving pod.

    AWS credentials (MinIO keys) must come from secretKeyRef, not from values files.
    """
    docs = _render_chart([VALUES_DEFAULT])

    serving = next(
        (
            d
            for d in docs
            if d.get("kind") == "Deployment"
            and d.get("metadata", {}).get("name") == "fastapi-serving"
        ),
        None,
    )
    assert serving is not None, "fastapi-serving Deployment not found"

    containers = serving["spec"]["template"]["spec"]["containers"]
    for c in containers:
        for entry in c.get("env") or []:
            value = str(entry.get("value") or "")
            for fragment in _FORBIDDEN_PLAINTEXT_IN_SERVING:
                assert fragment not in value, (
                    f"Hardcoded credential fragment '{fragment}' found in env var "
                    f"'{entry['name']}' of serving container '{c['name']}'.\n"
                    f"Use a secretKeyRef instead."
                )


# ── KEDA ScaledObject target ──────────────────────────────────────────────────


def test_keda_scaledobject_targets_existing_deployment():
    """The ScaledObject must reference a Deployment that exists in the same chart.

    A stale or renamed scaleTargetRef silently disables autoscaling — KEDA
    creates the ScaledObject but never adjusts replica counts.
    """
    docs = _render_chart([VALUES_DEFAULT])

    scaled_objects = [d for d in docs if d.get("kind") == "ScaledObject"]
    assert scaled_objects, (
        "No ScaledObject found in chart output. Is autoscaling.enabled set to true in values.yaml?"
    )

    deployment_names = {d["metadata"]["name"] for d in docs if d.get("kind") == "Deployment"}

    for so in scaled_objects:
        target = so["spec"]["scaleTargetRef"]["name"]
        assert target in deployment_names, (
            f"ScaledObject '{so['metadata']['name']}' targets Deployment '{target}' "
            f"but that Deployment does not exist in the chart.\n"
            f"Existing Deployments: {sorted(deployment_names)}"
        )


# ── Service selector / Deployment label consistency ───────────────────────────


def test_service_selectors_match_a_deployment():
    """Each Service selector must match the pod labels of at least one Deployment.

    A mismatch means the Service routes to zero pods — requests hang or return
    connection refused.
    """
    docs = _render_chart([VALUES_DEFAULT])

    pod_label_sets: list[tuple[str, dict]] = []
    for doc in docs:
        if doc.get("kind") == "Deployment":
            labels = doc.get("spec", {}).get("template", {}).get("metadata", {}).get("labels") or {}
            pod_label_sets.append((doc["metadata"]["name"], labels))

    for doc in docs:
        if doc.get("kind") != "Service":
            continue
        selector = (doc.get("spec") or {}).get("selector") or {}
        if not selector:
            continue  # Headless or externally-managed Service

        matched = any(
            all(pod_labels.get(k) == v for k, v in selector.items())
            for _, pod_labels in pod_label_sets
        )
        assert matched, (
            f"Service '{doc['metadata']['name']}' selector {selector} does not match "
            f"pod labels of any Deployment.\n"
            f"Deployment pod label sets: {pod_label_sets}"
        )
