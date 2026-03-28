# tests/unit/test_serving_api.py
"""
Unit tests for the FastAPI serving endpoints (serving/main.py).

Strategy
--------
- conftest.py sets the required env vars before this module is imported, so
  the module-level require_env() calls succeed.
- DATA_CONTROLLER_DB_URL is NOT set, so ServingDataController degrades
  gracefully (no-op writes) — no Postgres needed.
- model_manager.load_from_mlflow is patched to return False immediately and
  asyncio.sleep is patched to an AsyncMock so the lifespan retry loop
  completes instantly.
- Individual tests that need a "ready" model use the ready_model fixture,
  which sets classifier_session (makes is_ready True) and patches predict().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import serving.main as _serving

_VALID_IMAGE = [[0.5] * 14 for _ in range(14)]

_MOCK_PREDICT_RESULT = {
    "embedding": [0.0] * 64,
    "prediction": 3,
    "confidence": 0.85,
    "prediction_distribution": [0.05] * 9 + [0.40],
    "logits": [0.0] * 10,
}


@pytest.fixture(scope="module")
def client():
    """
    TestClient whose lifespan completes instantly without any MLflow calls.

    scope="module" so the app is only started once for all tests in this file.
    The module-level model_manager instance is shared; individual tests use
    the ready_model fixture to temporarily switch it to a loaded state.
    """
    with (
        patch.object(_serving.model_manager, "load_from_mlflow", return_value=False),
        patch("serving.main.asyncio.sleep", new_callable=AsyncMock),
        TestClient(_serving.app) as c,
    ):
        yield c


@pytest.fixture
def ready_model():
    """
    Temporarily configure model_manager as if a model were loaded.

    Sets classifier_session (which makes is_ready return True) and patches
    the predict() method to return a deterministic fake result.
    """
    original_session = _serving.model_manager.classifier_session
    original_version = _serving.model_manager.model_version

    _serving.model_manager.classifier_session = MagicMock()
    _serving.model_manager.model_version = "mock-run-id"
    _serving.model_manager.class_gaussians = None

    with patch.object(_serving.model_manager, "predict", return_value=_MOCK_PREDICT_RESULT):
        yield

    _serving.model_manager.classifier_session = original_session
    _serving.model_manager.model_version = original_version


# ── /health ───────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_503_when_model_not_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 503

    def test_response_has_required_fields(self, client):
        body = client.get("/health").json()
        assert "status" in body
        assert "model_loaded" in body
        assert "uptime_seconds" in body

    def test_model_loaded_false_when_not_ready(self, client):
        body = client.get("/health").json()
        assert body["model_loaded"] is False
        assert body["status"] == "unhealthy"

    def test_200_and_loaded_true_when_model_ready(self, client, ready_model):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["status"] == "healthy"
        assert body["model_version"] == "mock-run-id"


# ── /predict ──────────────────────────────────────────────────────────────────


class TestPredictEndpointModelNotReady:
    def test_503_before_model_loads(self, client):
        resp = client.post("/predict", json={"image": _VALID_IMAGE})
        assert resp.status_code == 503


class TestPredictEndpointValidRequests:
    def test_200_with_valid_image(self, client, ready_model):
        resp = client.post("/predict", json={"image": _VALID_IMAGE})
        assert resp.status_code == 200

    def test_response_contains_prediction_fields(self, client, ready_model):
        body = client.post("/predict", json={"image": _VALID_IMAGE}).json()
        assert body["prediction"] == 3
        assert body["confidence"] == pytest.approx(0.85)
        assert body["model_version"] == "mock-run-id"
        assert "uuid" in body

    def test_uuid_passthrough(self, client, ready_model):
        """UUID supplied in request is echoed back unchanged."""
        uid = "12345678-1234-5678-1234-567812345678"
        resp = client.post("/predict", json={"image": _VALID_IMAGE, "uuid": uid})
        assert resp.status_code == 200
        assert resp.json()["uuid"] == uid

    def test_uuid_auto_generated_when_absent(self, client, ready_model):
        """Two requests without a UUID get distinct auto-generated UUIDs."""
        r1 = client.post("/predict", json={"image": _VALID_IMAGE}).json()
        r2 = client.post("/predict", json={"image": _VALID_IMAGE}).json()
        assert r1["uuid"] != r2["uuid"]

    def test_black_image_accepted(self, client, ready_model):
        black = [[0.0] * 14 for _ in range(14)]
        assert client.post("/predict", json={"image": black}).status_code == 200

    def test_white_image_accepted(self, client, ready_model):
        white = [[1.0] * 14 for _ in range(14)]
        assert client.post("/predict", json={"image": white}).status_code == 200


class TestPredictEndpointInvalidRequests:
    def test_422_for_too_few_rows(self, client, ready_model):
        bad = [[0.5] * 14 for _ in range(7)]
        resp = client.post("/predict", json={"image": bad})
        assert resp.status_code == 422

    def test_422_for_too_many_rows(self, client, ready_model):
        bad = [[0.5] * 14 for _ in range(20)]
        resp = client.post("/predict", json={"image": bad})
        assert resp.status_code == 422

    def test_422_for_wrong_column_count(self, client, ready_model):
        bad = [[0.5] * 14 for _ in range(14)]
        bad[5] = [0.5] * 8
        resp = client.post("/predict", json={"image": bad})
        assert resp.status_code == 422

    def test_422_for_value_out_of_range(self, client, ready_model):
        bad = [[0.5] * 14 for _ in range(14)]
        bad[0][0] = 2.0
        resp = client.post("/predict", json={"image": bad})
        assert resp.status_code == 422

    def test_422_for_negative_value(self, client, ready_model):
        bad = [[0.5] * 14 for _ in range(14)]
        bad[13][13] = -0.5
        resp = client.post("/predict", json={"image": bad})
        assert resp.status_code == 422

    def test_422_for_missing_image_field(self, client, ready_model):
        resp = client.post("/predict", json={"uuid": "12345678-1234-5678-1234-567812345678"})
        # FastAPI returns 422 for missing required field
        assert resp.status_code == 422
