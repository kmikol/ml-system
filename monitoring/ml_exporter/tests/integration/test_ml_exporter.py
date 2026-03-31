"""Integration tests for the ML exporter service.

Requires a running ml_exporter instance with DB and MLflow reachable:
  make k3d.redeploy  (or docker-compose up ml_exporter)

Endpoint: http://localhost:8001
"""

import httpx

BASE = "http://localhost:8001"


def test_health_returns_200():
    r = httpx.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200


def test_health_has_required_fields():
    body = httpx.get(f"{BASE}/health", timeout=10).json()
    for field in [
        "status",
        "model_run_id",
        "reference_loaded",
        "last_poll_age_seconds",
        "poll_interval",
        "window_seconds",
    ]:
        assert field in body, f"Missing field: {field}"


def test_health_status_is_healthy():
    body = httpx.get(f"{BASE}/health", timeout=10).json()
    assert body["status"] == "healthy"


def test_health_poll_interval_is_positive():
    body = httpx.get(f"{BASE}/health", timeout=10).json()
    assert body["poll_interval"] > 0
    assert body["window_seconds"] > 0


def test_metrics_returns_200():
    r = httpx.get(f"{BASE}/metrics", timeout=10)
    assert r.status_code == 200


def test_metrics_content_type_is_prometheus():
    r = httpx.get(f"{BASE}/metrics", timeout=10)
    assert "text/plain" in r.headers["content-type"]


def test_metrics_body_contains_expected_metric_names():
    body = httpx.get(f"{BASE}/metrics", timeout=10).text
    for metric in [
        "drift_window_sample_count",
        "drift_window_class_count",
        "drift_window_class_freq",
        "drift_window_confidence_mean",
        "drift_psi_class_distribution",
        "drift_last_poll_age_seconds",
        "annotation_annotated_count",
    ]:
        assert metric in body, f"Metric {metric!r} not found in /metrics output"
