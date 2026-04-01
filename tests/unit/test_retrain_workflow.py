from __future__ import annotations

from pathlib import Path


def _workflow_text() -> str:
    return Path("k8s/argo/workflows/retrain.yaml").read_text(encoding="utf-8")


def test_retrain_uses_secret_refs_for_lakefs_credentials() -> None:
    text = _workflow_text()

    # Three steps use lakeFS credentials: integrate-annotations, train, evaluate-and-promote.
    assert text.count("name: LAKEFS_ACCESS_KEY_ID") >= 3
    assert text.count("name: LAKEFS_SECRET_ACCESS_KEY") >= 3
    assert text.count("name: ml-system-secrets") >= 6
    assert text.count("key: lakefs-access-key-id") >= 3
    assert text.count("key: lakefs-secret-access-key") >= 3

    # Guard against accidental hardcoding of known sample credentials.
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in text


def test_integrate_annotations_waits_for_lakefs() -> None:
    text = _workflow_text()

    assert "- name: integrate-annotations" in text
    assert "- name: wait-for-lakefs" in text
    assert "http://lakefs.ml-system.svc.cluster.local:8000/health" in text
