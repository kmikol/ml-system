from __future__ import annotations

from pathlib import Path


def _seed_data_section() -> str:
    workflow = Path("argo/workflows/initialize-workflow.yaml").read_text(encoding="utf-8")
    start = workflow.index("    - name: seed-data\n      initContainers:")
    end = workflow.index("    # ── Step 2: verify-data")
    return workflow[start:end]


def test_seed_data_step_includes_lakefs_environment() -> None:
    section = _seed_data_section()

    assert "name: LAKEFS_ENDPOINT_URL" in section
    assert "http://lakefs.ml-system.svc.cluster.local:8000" in section
    assert "name: LAKEFS_ACCESS_KEY_ID" in section
    assert "name: ml-system-secrets" in section
    assert "key: lakefs-access-key-id" in section
    assert "name: LAKEFS_SECRET_ACCESS_KEY" in section
    assert "key: lakefs-secret-access-key" in section
    assert "name: LAKEFS_REPO" in section
    assert 'value: "ml-system-datasets"' in section
    assert "AKIAIOSFODNN7EXAMPLE" not in section
    assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in section


def test_seed_data_waits_for_lakefs_readiness() -> None:
    section = _seed_data_section()

    assert "name: wait-for-lakefs" in section
    assert "http://lakefs.ml-system.svc.cluster.local:8000/health" in section
