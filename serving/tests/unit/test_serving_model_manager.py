# serving/tests/unit/test_serving_model_manager.py
"""
Unit tests for serving/main.py::ModelManager internals.

Tests ModelManager.predict(), load_from_mlflow(), is_ready, and the
Mahalanobis scoring path in the /predict endpoint.

All ONNX sessions and external controllers are mocked — no MLflow or GPU needed.
conftest.py sets the required env vars before this module is imported.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.special import softmax

import serving.main as _serving
from shared.model_artifact_controller import ModelArtifactError

INPUT_DIM = 196
EMBEDDING_DIM = 64
NUM_CLASSES = 10

_VALID_IMAGE = [[0.5] * 14 for _ in range(14)]


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_logits(prediction: int = 3) -> np.ndarray:
    """Return a (1, 10) logits array where class `prediction` is dominant."""
    logits = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    logits[0, prediction] = 10.0
    return logits


def _make_embedding() -> np.ndarray:
    return np.random.randn(1, EMBEDDING_DIM).astype(np.float32)


def _mock_sessions(prediction: int = 3):
    """Return a unified model session mock."""
    model_sess = MagicMock()
    model_sess.run.return_value = [_make_logits(prediction), _make_embedding()]
    return model_sess


@pytest.fixture
def fresh_manager() -> _serving.ModelManager:
    """Fresh ModelManager with a mocked artifact controller."""
    with patch("serving.main.ModelArtifactController"):
        manager = _serving.ModelManager()
    return manager


@pytest.fixture
def ready_manager(fresh_manager) -> _serving.ModelManager:
    """ModelManager with unified model session loaded."""
    model_sess = _mock_sessions(prediction=3)
    fresh_manager.model_session = model_sess
    fresh_manager.model_version = "mock-run-id"
    fresh_manager.class_gaussians = None
    return fresh_manager


# ── ModelManager.is_ready ─────────────────────────────────────────────────────


class TestIsReady:
    def test_false_before_load(self, fresh_manager):
        assert fresh_manager.is_ready is False

    def test_true_after_model_session_set(self, fresh_manager):
        fresh_manager.model_session = MagicMock()
        assert fresh_manager.is_ready is True

    def test_false_after_session_cleared(self, ready_manager):
        ready_manager.model_session = None
        assert ready_manager.is_ready is False


# ── ModelManager.predict ──────────────────────────────────────────────────────


class TestModelManagerPredict:
    def test_raises_when_not_loaded(self, fresh_manager):
        features = np.zeros((1, INPUT_DIM), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            fresh_manager.predict(features)

    def test_returns_required_keys(self, ready_manager):
        features = np.zeros((1, INPUT_DIM), dtype=np.float32)
        result = ready_manager.predict(features)
        assert set(result.keys()) >= {
            "logits",
            "embedding",
            "prediction",
            "confidence",
            "prediction_distribution",
        }

    def test_prediction_is_argmax(self, fresh_manager):
        for predicted_class in [0, 5, 9]:
            model_sess = _mock_sessions(prediction=predicted_class)
            fresh_manager.model_session = model_sess
            result = fresh_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
            assert result["prediction"] == predicted_class

    def test_confidence_matches_softmax(self, ready_manager):
        logits = _make_logits(prediction=3)
        ready_manager.model_session.run.return_value = [logits, _make_embedding()]
        result = ready_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
        expected_probs = softmax(logits[0])
        assert result["confidence"] == pytest.approx(float(expected_probs[3]), abs=1e-5)

    def test_embedding_length(self, ready_manager):
        result = ready_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
        assert len(result["embedding"]) == EMBEDDING_DIM

    def test_prediction_distribution_length(self, ready_manager):
        result = ready_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
        assert len(result["prediction_distribution"]) == NUM_CLASSES

    def test_prediction_distribution_sums_to_one(self, ready_manager):
        result = ready_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
        assert abs(sum(result["prediction_distribution"]) - 1.0) < 1e-5

    def test_confidence_within_distribution(self, ready_manager):
        result = ready_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
        pred_class = result["prediction"]
        assert result["confidence"] == pytest.approx(
            result["prediction_distribution"][pred_class], abs=1e-5
        )

    def test_passes_features_as_float32_to_sessions(self, ready_manager):
        features = np.ones((1, INPUT_DIM), dtype=np.float64)
        ready_manager.predict(features)
        call_kwargs = ready_manager.model_session.run.call_args[0][1]
        assert call_kwargs["features"].dtype == np.float32


# ── ModelManager.predict shape validation ─────────────────────────────────────


class TestPredictShapeValidation:
    """Test shape validation in predict() to prevent IndexError."""

    def test_raises_on_empty_logits_batch(self, fresh_manager):
        """Empty logits batch should raise ValueError."""
        model_sess = MagicMock()
        # Return empty batch: (0, 10) shape
        model_sess.run.return_value = [
            np.zeros((0, NUM_CLASSES), dtype=np.float32),
            np.zeros((1, EMBEDDING_DIM), dtype=np.float32),
        ]
        fresh_manager.model_session = model_sess
        fresh_manager.model_version = "test-run"

        features = np.zeros((1, INPUT_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid logits shape.*batch_size >= 1"):
            fresh_manager.predict(features)

    def test_raises_on_empty_embedding_batch(self, fresh_manager):
        """Empty embedding batch should raise ValueError."""
        model_sess = MagicMock()
        # Return empty embedding batch: (0, 64) shape
        model_sess.run.return_value = [
            np.zeros((1, NUM_CLASSES), dtype=np.float32),
            np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        ]
        fresh_manager.model_session = model_sess
        fresh_manager.model_version = "test-run"

        features = np.zeros((1, INPUT_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid embedding shape.*batch_size >= 1"):
            fresh_manager.predict(features)

    def test_raises_on_1d_logits(self, fresh_manager):
        """1D logits array should raise ValueError."""
        model_sess = MagicMock()
        # Return 1D array instead of 2D
        model_sess.run.return_value = [
            np.zeros((NUM_CLASSES,), dtype=np.float32),
            np.zeros((1, EMBEDDING_DIM), dtype=np.float32),
        ]
        fresh_manager.model_session = model_sess
        fresh_manager.model_version = "test-run"

        features = np.zeros((1, INPUT_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid logits shape"):
            fresh_manager.predict(features)

    def test_raises_on_1d_embedding(self, fresh_manager):
        """1D embedding array should raise ValueError."""
        model_sess = MagicMock()
        # Return 1D embedding instead of 2D
        model_sess.run.return_value = [
            np.zeros((1, NUM_CLASSES), dtype=np.float32),
            np.zeros((EMBEDDING_DIM,), dtype=np.float32),
        ]
        fresh_manager.model_session = model_sess
        fresh_manager.model_version = "test-run"

        features = np.zeros((1, INPUT_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid embedding shape"):
            fresh_manager.predict(features)

    def test_raises_on_3d_logits(self, fresh_manager):
        """3D logits array should raise ValueError."""
        model_sess = MagicMock()
        # Return 3D array
        model_sess.run.return_value = [
            np.zeros((1, 1, NUM_CLASSES), dtype=np.float32),
            np.zeros((1, EMBEDDING_DIM), dtype=np.float32),
        ]
        fresh_manager.model_session = model_sess
        fresh_manager.model_version = "test-run"

        features = np.zeros((1, INPUT_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid logits shape"):
            fresh_manager.predict(features)

    def test_accepts_valid_shapes(self, fresh_manager):
        """Valid (1, num_classes) and (1, embedding_dim) shapes should work."""
        model_sess = MagicMock()
        model_sess.run.return_value = [
            _make_logits(prediction=5),
            _make_embedding(),
        ]
        fresh_manager.model_session = model_sess
        fresh_manager.model_version = "test-run"

        features = np.zeros((1, INPUT_DIM), dtype=np.float32)
        result = fresh_manager.predict(features)
        # Should not raise and should return valid result
        assert result["prediction"] == 5
        assert len(result["embedding"]) == EMBEDDING_DIM


# ── ModelManager.load_from_mlflow ─────────────────────────────────────────────


class TestLoadFromMlflow:
    def test_returns_false_when_get_run_id_raises(self, fresh_manager):
        fresh_manager._controller.get_production_run_id.side_effect = ModelArtifactError(
            "MLflow down"
        )
        result = fresh_manager.load_from_mlflow()
        assert result is False

    def test_returns_true_on_cache_hit(self, fresh_manager):
        fresh_manager.model_version = "run-42"
        fresh_manager._controller.get_production_run_id.return_value = "run-42"
        result = fresh_manager.load_from_mlflow()
        assert result is True

    def test_returns_false_when_download_raises(self, fresh_manager):
        fresh_manager._controller.get_production_run_id.return_value = "new-run"
        fresh_manager._controller.download_serving_bundle.side_effect = ModelArtifactError(
            "Download failed"
        )
        result = fresh_manager.load_from_mlflow()
        assert result is False

    def test_sets_model_version_on_success(self, fresh_manager, tmp_path):
        model_sess = _mock_sessions()
        fresh_manager._controller.get_production_run_id.return_value = "run-99"
        fresh_manager._controller.download_serving_bundle.return_value = (
            str(tmp_path / "model.onnx"),
            None,
        )
        with patch("serving.main.ort.InferenceSession", return_value=model_sess):
            result = fresh_manager.load_from_mlflow()
        assert result is True
        assert fresh_manager.model_version == "run-99"

    def test_sets_class_gaussians_to_none_when_absent(self, fresh_manager, tmp_path):
        model_sess = _mock_sessions()
        fresh_manager._controller.get_production_run_id.return_value = "run-1"
        fresh_manager._controller.download_serving_bundle.return_value = (
            str(tmp_path),
            None,  # raw_gaussians=None
        )
        with patch("serving.main.ort.InferenceSession", return_value=model_sess):
            fresh_manager.load_from_mlflow()
        assert fresh_manager.class_gaussians is None

    def test_loads_valid_gaussians(self, fresh_manager, tmp_path):
        model_sess = _mock_sessions()
        raw_gaussians = {
            "classes": {
                "3": {
                    "mean": [0.0] * EMBEDDING_DIM,
                    "precision": [
                        [1.0 if i == j else 0.0 for j in range(EMBEDDING_DIM)]
                        for i in range(EMBEDDING_DIM)
                    ],
                }
            }
        }
        fresh_manager._controller.get_production_run_id.return_value = "run-1"
        fresh_manager._controller.download_serving_bundle.return_value = (
            str(tmp_path),
            raw_gaussians,
        )
        with patch("serving.main.ort.InferenceSession", return_value=model_sess):
            fresh_manager.load_from_mlflow()
        assert fresh_manager.class_gaussians is not None
        assert "3" in fresh_manager.class_gaussians
        g = fresh_manager.class_gaussians["3"]
        assert isinstance(g["mean"], np.ndarray)
        assert isinstance(g["precision"], np.ndarray)
        assert g["mean"].shape == (EMBEDDING_DIM,)
        assert g["precision"].shape == (EMBEDDING_DIM, EMBEDDING_DIM)

    def test_invalid_gaussians_payload_sets_none(self, fresh_manager, tmp_path):
        model_sess = _mock_sessions()
        # Missing 'classes' key — invalid structure
        raw_gaussians = {"bad_key": {}}
        fresh_manager._controller.get_production_run_id.return_value = "run-1"
        fresh_manager._controller.download_serving_bundle.return_value = (
            str(tmp_path),
            raw_gaussians,
        )
        with patch("serving.main.ort.InferenceSession", return_value=model_sess):
            fresh_manager.load_from_mlflow()
        # Invalid payload → scoring disabled (class_gaussians is None)
        assert fresh_manager.class_gaussians is None


# ── Mahalanobis scoring in /predict endpoint ──────────────────────────────────


@pytest.fixture(scope="module")
def client():
    with (
        patch.object(_serving.model_manager, "load_from_mlflow", return_value=False),
        patch("serving.main.asyncio.sleep", new_callable=AsyncMock),
        TestClient(_serving.app) as c,
    ):
        yield c


@pytest.fixture
def ready_model_with_gaussians():
    """Set model_manager to a ready state WITH class Gaussians for class 3."""
    original_session = _serving.model_manager.model_session
    original_version = _serving.model_manager.model_version
    original_gaussians = _serving.model_manager.class_gaussians

    _serving.model_manager.model_session = MagicMock()
    _serving.model_manager.model_version = "mock-run-id"

    mean = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    precision = np.eye(EMBEDDING_DIM, dtype=np.float64)
    _serving.model_manager.class_gaussians = {"3": {"mean": mean, "precision": precision}}

    mock_result = {
        "embedding": [0.0] * EMBEDDING_DIM,
        "prediction": 3,
        "confidence": 0.85,
        "prediction_distribution": [0.05] * 9 + [0.40],
        "logits": [0.0] * NUM_CLASSES,
    }
    with patch.object(_serving.model_manager, "predict", return_value=mock_result):
        yield

    _serving.model_manager.model_session = original_session
    _serving.model_manager.model_version = original_version
    _serving.model_manager.class_gaussians = original_gaussians


@pytest.fixture
def ready_model_gaussians_missing_class():
    """Gaussians loaded but missing the predicted class key."""
    original_session = _serving.model_manager.model_session
    original_version = _serving.model_manager.model_version
    original_gaussians = _serving.model_manager.class_gaussians

    _serving.model_manager.model_session = MagicMock()
    _serving.model_manager.model_version = "mock-run-id"
    # No key "3" — the predicted class
    _serving.model_manager.class_gaussians = {
        "9": {"mean": np.zeros(EMBEDDING_DIM), "precision": np.eye(EMBEDDING_DIM)}
    }

    mock_result = {
        "embedding": [0.0] * EMBEDDING_DIM,
        "prediction": 3,
        "confidence": 0.85,
        "prediction_distribution": [0.05] * 9 + [0.40],
        "logits": [0.0] * NUM_CLASSES,
    }
    with patch.object(_serving.model_manager, "predict", return_value=mock_result):
        yield

    _serving.model_manager.model_session = original_session
    _serving.model_manager.model_version = original_version
    _serving.model_manager.class_gaussians = original_gaussians


class TestMahalanobisScoring:
    def test_returns_200_with_valid_gaussians(self, client, ready_model_with_gaussians):
        resp = client.post("/predict", json={"image": _VALID_IMAGE})
        assert resp.status_code == 200

    def test_returns_200_when_gaussians_missing_class(
        self, client, ready_model_gaussians_missing_class
    ):
        """Scoring gracefully skips when the predicted class has no Gaussian."""
        resp = client.post("/predict", json={"image": _VALID_IMAGE})
        assert resp.status_code == 200

    def test_response_structure_unchanged_with_gaussians(self, client, ready_model_with_gaussians):
        body = client.post("/predict", json={"image": _VALID_IMAGE}).json()
        assert "prediction" in body
        assert "confidence" in body
        assert "uuid" in body
