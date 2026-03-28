# tests/unit/test_ml_exporter_app.py
"""
Unit tests for monitoring/ml_exporter/main.py — the untested portions:
  - ExporterConfig.from_env()
  - PrometheusEmitter (emit, update_poll_age, generate_metrics)
  - /health endpoint
  - /metrics endpoint

No DB, MLflow, or background threads are started. The FastAPI app is tested
via TestClient with manually wired app.state so the lifespan is bypassed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from monitoring.ml_exporter.main import (
    _PSI_SENTINEL,
    ExporterConfig,
    PrometheusEmitter,
    WindowMetrics,
    app,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_metrics(n: int = 50, psi: float | None = 0.05) -> WindowMetrics:
    counts = dict.fromkeys(range(10), n // 10)
    freqs = dict.fromkeys(range(10), 0.1)
    return WindowMetrics(
        n=n,
        class_counts=counts,
        class_freqs=freqs,
        confidence_mean=0.85,
        psi=psi,
    )


# ── ExporterConfig.from_env ───────────────────────────────────────────────────


class TestExporterConfigFromEnv:
    def test_reads_all_env_vars(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "my-model")
        monkeypatch.setenv("MODEL_STAGE", "Staging")
        monkeypatch.setenv("DRIFT_POLL_INTERVAL", "30")
        monkeypatch.setenv("DRIFT_WINDOW_SECONDS", "7200")
        monkeypatch.setenv("DRIFT_MIN_SAMPLES", "50")
        cfg = ExporterConfig.from_env()
        assert cfg.model_name == "my-model"
        assert cfg.model_stage == "Staging"
        assert cfg.poll_interval == 30
        assert cfg.window_seconds == 7200
        assert cfg.min_samples == 50

    def test_exits_when_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("DRIFT_POLL_INTERVAL", raising=False)
        monkeypatch.setenv("MODEL_NAME", "m")
        monkeypatch.setenv("MODEL_STAGE", "Production")
        monkeypatch.setenv("DRIFT_WINDOW_SECONDS", "3600")
        monkeypatch.setenv("DRIFT_MIN_SAMPLES", "30")
        with pytest.raises(SystemExit):
            ExporterConfig.from_env()


# ── PrometheusEmitter ─────────────────────────────────────────────────────────


class TestPrometheusEmitter:
    @pytest.fixture
    def emitter(self) -> PrometheusEmitter:
        return PrometheusEmitter()

    def test_generate_metrics_returns_bytes_and_content_type(self, emitter):
        data, content_type = emitter.generate_metrics()
        assert isinstance(data, bytes)
        assert "text/plain" in content_type

    def test_emit_sets_sample_count(self, emitter):
        emitter.emit(_make_metrics(n=42), annotated_count=0)
        data, _ = emitter.generate_metrics()
        assert b"drift_window_sample_count 42.0" in data

    def test_emit_includes_all_metric_names(self, emitter):
        emitter.emit(_make_metrics(), annotated_count=5)
        data, _ = emitter.generate_metrics()
        for name in [
            b"drift_window_sample_count",
            b"drift_window_class_count",
            b"drift_window_class_freq",
            b"drift_window_confidence_mean",
            b"drift_psi_class_distribution",
            b"annotation_annotated_count",
        ]:
            assert name in data, f"Metric {name!r} missing from output"

    def test_emit_psi_sentinel_when_psi_none(self, emitter):
        emitter.emit(_make_metrics(psi=None), annotated_count=0)
        data, _ = emitter.generate_metrics()
        sentinel_bytes = str(_PSI_SENTINEL).encode()
        assert sentinel_bytes in data

    def test_emit_psi_actual_value(self, emitter):
        emitter.emit(_make_metrics(psi=0.123), annotated_count=0)
        data, _ = emitter.generate_metrics()
        assert b"0.123" in data

    def test_emit_annotated_count(self, emitter):
        emitter.emit(_make_metrics(), annotated_count=17)
        data, _ = emitter.generate_metrics()
        assert b"annotation_annotated_count 17.0" in data

    def test_update_poll_age(self, emitter):
        emitter.update_poll_age(42.5)
        data, _ = emitter.generate_metrics()
        assert b"drift_last_poll_age_seconds 42.5" in data

    def test_confidence_mean_emitted(self, emitter):
        emitter.emit(_make_metrics(n=50), annotated_count=0)
        data, _ = emitter.generate_metrics()
        assert b"drift_window_confidence_mean 0.85" in data

    def test_class_labels_present_in_output(self, emitter):
        emitter.emit(_make_metrics(), annotated_count=0)
        data, _ = emitter.generate_metrics()
        for c in range(10):
            assert f'class_label="{c}"'.encode() in data


# ── /health endpoint ──────────────────────────────────────────────────────────


@pytest.fixture
def test_client_with_state(monkeypatch):
    """
    TestClient with manually wired app.state.

    The ml_exporter lifespan calls ExporterConfig.from_env() (needs env vars),
    creates real data/artifact controllers (need DB/MLflow), and spawns background
    threads. We set the minimum required env vars, mock out all external calls,
    and overwrite app.state with our test objects immediately after startup.
    """
    monkeypatch.setenv("DRIFT_POLL_INTERVAL", "60")
    monkeypatch.setenv("DRIFT_WINDOW_SECONDS", "3600")
    monkeypatch.setenv("DRIFT_MIN_SAMPLES", "30")
    # MODEL_NAME / MODEL_STAGE / MLFLOW_TRACKING_URI already set by conftest.py

    mock_poller = MagicMock()
    mock_poller.current_run_id.return_value = "run-42"
    mock_poller.reference_loaded.return_value = True
    mock_poller.poll_age.return_value = 5.3

    mock_emitter = PrometheusEmitter()
    mock_config = ExporterConfig(
        model_name="test-model",
        model_stage="Production",
        poll_interval=60,
        window_seconds=3600,
        min_samples=30,
    )

    with (
        patch("monitoring.ml_exporter.main.DriftDataController"),
        patch("monitoring.ml_exporter.main.ModelArtifactController"),
        patch("monitoring.ml_exporter.main.threading.Thread"),
        TestClient(app, raise_server_exceptions=True) as client,
    ):
        # Overwrite state set by lifespan with our controlled test objects
        client.app.state.poller = mock_poller
        client.app.state.emitter = mock_emitter
        client.app.state.config = mock_config
        yield client, mock_poller


class TestHealthEndpoint:
    def test_returns_200(self, test_client_with_state):
        client, _ = test_client_with_state
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_required_fields(self, test_client_with_state):
        client, _ = test_client_with_state
        body = client.get("/health").json()
        for field in [
            "status",
            "model_run_id",
            "reference_loaded",
            "last_poll_age_seconds",
            "poll_interval",
            "window_seconds",
        ]:
            assert field in body, f"Missing field: {field}"

    def test_status_is_healthy(self, test_client_with_state):
        client, _ = test_client_with_state
        body = client.get("/health").json()
        assert body["status"] == "healthy"

    def test_reflects_poller_state(self, test_client_with_state):
        client, mock_poller = test_client_with_state
        body = client.get("/health").json()
        assert body["model_run_id"] == "run-42"
        assert body["reference_loaded"] is True
        assert body["last_poll_age_seconds"] == pytest.approx(5.3, abs=0.1)

    def test_poll_age_none_when_never_polled(self, test_client_with_state):
        client, mock_poller = test_client_with_state
        mock_poller.poll_age.return_value = None
        body = client.get("/health").json()
        assert body["last_poll_age_seconds"] is None

    def test_config_values_in_response(self, test_client_with_state):
        client, _ = test_client_with_state
        body = client.get("/health").json()
        assert body["poll_interval"] == 60
        assert body["window_seconds"] == 3600


# ── /metrics endpoint ─────────────────────────────────────────────────────────


class TestMetricsEndpoint:
    def test_returns_200(self, test_client_with_state):
        client, _ = test_client_with_state
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_content_type_is_prometheus(self, test_client_with_state):
        client, _ = test_client_with_state
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_body_is_non_empty(self, test_client_with_state):
        client, _ = test_client_with_state
        resp = client.get("/metrics")
        assert len(resp.content) > 0

    def test_body_contains_metric_after_emit(self, test_client_with_state):
        client, _ = test_client_with_state
        # Emit some data into the emitter
        client.app.state.emitter.emit(_make_metrics(n=100), annotated_count=3)
        resp = client.get("/metrics")
        assert b"drift_window_sample_count" in resp.content
