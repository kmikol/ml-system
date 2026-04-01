# tests/unit/test_serving_model_manager.py
"""
Unit tests for serving/main.py::ModelManager internals.

Tests ModelManager.predict(), load_from_registry(), is_ready, and the
Mahalanobis scoring path in the /predict endpoint.

All ONNX sessions and external controllers are mocked — no MLflow or GPU needed.
conftest.py sets the required env vars before this module is imported.
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.special import softmax

import serving.main as _serving
from shared.model_artifact_controller import (
    ClassGaussian,
    ClassGaussians,
    ModelArtifactError,
    ServingBundle,
)

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
    """Fresh ModelManager with a mocked ModelStore."""
    with patch("serving.main.ModelStore"):
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


# ── ModelManager.load_from_registry ───────────────────────────────────────────


class TestLoadFromRegistry:
    def test_returns_false_when_get_version_id_raises(self, fresh_manager):
        fresh_manager._store.get_current_version_id.side_effect = ModelArtifactError(
            "Registry down"
        )
        result = fresh_manager.load_from_registry()
        assert result is False

    def test_returns_true_on_cache_hit(self, fresh_manager):
        fresh_manager.model_version = "run-42"
        fresh_manager._store.get_current_version_id.return_value = "run-42"
        result = fresh_manager.load_from_registry()
        assert result is True

    def test_returns_false_when_download_raises(self, fresh_manager):
        fresh_manager._store.get_current_version_id.return_value = "new-run"
        fresh_manager._store.get_serving_bundle.side_effect = ModelArtifactError("Download failed")
        result = fresh_manager.load_from_registry()
        assert result is False

    def test_sets_model_version_on_success(self, fresh_manager, tmp_path):
        model_sess = _mock_sessions()
        fresh_manager._store.get_current_version_id.return_value = "run-99"
        fresh_manager._store.get_serving_bundle.return_value = ServingBundle(
            model_path=str(tmp_path / "model.onnx"),
            class_gaussians=None,
        )
        with patch("serving.main.ort.InferenceSession", return_value=model_sess):
            result = fresh_manager.load_from_registry()
        assert result is True
        assert fresh_manager.model_version == "run-99"

    def test_sets_class_gaussians_to_none_when_absent(self, fresh_manager, tmp_path):
        model_sess = _mock_sessions()
        fresh_manager._store.get_current_version_id.return_value = "run-1"
        fresh_manager._store.get_serving_bundle.return_value = ServingBundle(
            model_path=str(tmp_path),
            class_gaussians=None,
        )
        with patch("serving.main.ort.InferenceSession", return_value=model_sess):
            fresh_manager.load_from_registry()
        assert fresh_manager.class_gaussians is None

    def test_loads_valid_gaussians(self, fresh_manager, tmp_path):
        model_sess = _mock_sessions()
        gaussians = ClassGaussians(
            classes={
                "3": ClassGaussian(
                    mean=[0.0] * EMBEDDING_DIM,
                    precision=[
                        [1.0 if i == j else 0.0 for j in range(EMBEDDING_DIM)]
                        for i in range(EMBEDDING_DIM)
                    ],
                    num_samples=100,
                )
            }
        )
        fresh_manager._store.get_current_version_id.return_value = "run-1"
        fresh_manager._store.get_serving_bundle.return_value = ServingBundle(
            model_path=str(tmp_path),
            class_gaussians=gaussians,
        )
        with patch("serving.main.ort.InferenceSession", return_value=model_sess):
            fresh_manager.load_from_registry()
        assert fresh_manager.class_gaussians is not None
        assert "3" in fresh_manager.class_gaussians
        g = fresh_manager.class_gaussians["3"]
        assert isinstance(g["mean"], np.ndarray)
        assert isinstance(g["precision"], np.ndarray)
        assert g["mean"].shape == (EMBEDDING_DIM,)
        assert g["precision"].shape == (EMBEDDING_DIM, EMBEDDING_DIM)


# ── Mahalanobis scoring in /predict endpoint ──────────────────────────────────


@pytest.fixture(scope="module")
def client():
    with (
        patch.object(_serving.model_manager, "load_from_registry", return_value=False),
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


# ── Thread-safety / concurrency tests ────────────────────────────────────────


class TestConcurrency:
    """Verify that predict() is safe under concurrent access and model reloads."""

    def test_concurrent_predictions_all_succeed(self, fresh_manager):
        """Multiple threads calling predict() simultaneously must all succeed."""
        model_sess = _mock_sessions(prediction=7)
        fresh_manager.model_session = model_sess
        fresh_manager.model_version = "test-run"

        errors: list[Exception] = []
        results: list[dict] = []
        lock = threading.Lock()

        def run_predict():
            try:
                result = fresh_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
                with lock:
                    results.append(result)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=run_predict) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent predict raised: {errors}"
        assert len(results) == 20
        assert all(r["prediction"] == 7 for r in results)

    def test_model_reload_during_inference_does_not_corrupt_result(self, fresh_manager):
        """Replacing the session while inference is in flight must not corrupt results.

        We simulate a slow inference (via a side-effect that briefly yields to
        the scheduler) and a concurrent load_from_registry()-style swap.  After
        both threads finish every result must be self-consistent — the
        prediction and confidence must always agree with the same session's
        logits.
        """
        barrier = threading.Barrier(2)

        v1_prediction = 3
        v2_prediction = 7

        v1_logits = _make_logits(v1_prediction)
        v2_logits = _make_logits(v2_prediction)

        def v1_run(output_names, feed):
            # Signal that inference has started, then wait for the swap thread
            # to be ready so we maximise the chance of overlap.
            barrier.wait(timeout=5)
            return [v1_logits, _make_embedding()]

        session_v1 = MagicMock()
        session_v1.run.side_effect = v1_run

        session_v2 = MagicMock()
        session_v2.run.return_value = [v2_logits, _make_embedding()]

        fresh_manager.model_session = session_v1
        fresh_manager.model_version = "v1"

        inference_result: list[dict] = []
        inference_errors: list[Exception] = []

        def run_inference():
            try:
                res = fresh_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
                inference_result.append(res)
            except Exception as exc:  # noqa: BLE001
                inference_errors.append(exc)

        def swap_session():
            # Wait until inference has captured its local reference, then swap.
            barrier.wait(timeout=5)
            with fresh_manager._lock:
                fresh_manager.model_session = session_v2
                fresh_manager.model_version = "v2"

        t_infer = threading.Thread(target=run_inference)
        t_swap = threading.Thread(target=swap_session)

        t_infer.start()
        t_swap.start()
        t_infer.join(timeout=5)
        t_swap.join(timeout=5)

        assert inference_errors == [], f"Inference raised: {inference_errors}"
        assert len(inference_result) == 1

        result = inference_result[0]
        # The prediction must correspond to one consistent session.
        # v1_prediction=3, v2_prediction=7 — must not be any other value.
        assert result["prediction"] in {v1_prediction, v2_prediction}
        # The confidence reported must match the distribution entry for that class.
        pred_class = result["prediction"]
        assert result["confidence"] == pytest.approx(
            result["prediction_distribution"][pred_class], abs=1e-5
        )

    def test_predict_raises_when_session_is_none_at_check_time(self, fresh_manager):
        """If model_session is None when predict() acquires the lock it must raise."""
        fresh_manager.model_session = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            fresh_manager.predict(np.zeros((1, INPUT_DIM), dtype=np.float32))
