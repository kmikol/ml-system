# tests/unit/test_training_pipeline.py
"""
Unit tests for training/main.py pure functions and main() orchestration.

Env vars (MODEL_NAME, TRAINING_*) are set by conftest.py before import.
External services (DatasetController, ModelArtifactController, pl.Trainer)
are fully mocked so no Postgres, MLflow, or real training is needed.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from shared.schemas.feature_schema import EMBEDDING_DIM, INPUT_DIM, NUM_CLASSES
from training.main import compute_reference_distributions, export_onnx, make_dataloaders
from training.model import Classifier

LR = 0.001


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_samples(n: int, label: int | None = None) -> list[dict]:
    """Create n fake dataset samples (14×14 images with random pixel values)."""
    rng = np.random.default_rng(0)
    samples = []
    for i in range(n):
        image = rng.random((14, 14)).tolist()
        lbl = label if label is not None else (i % NUM_CLASSES)
        samples.append({"image": image, "label": lbl})
    return samples


@pytest.fixture
def small_model() -> Classifier:
    m = Classifier(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES, lr=LR)
    m.eval()
    return m


# ── make_dataloaders ──────────────────────────────────────────────────────────


class TestMakeDataloaders:
    def test_returns_two_dataloaders(self):
        train_samples = _make_samples(20)
        val_samples = _make_samples(10)
        train_dl, val_dl = make_dataloaders(train_samples, val_samples, batch_size=8)
        assert train_dl is not None
        assert val_dl is not None

    def test_x_shape_is_flattened(self):
        samples = _make_samples(16)
        dl, _ = make_dataloaders(samples, _make_samples(4), batch_size=16)
        x, y = next(iter(dl))
        assert x.shape == (16, INPUT_DIM)  # 14*14 = 196

    def test_y_shape_and_dtype(self):
        samples = _make_samples(8)
        dl, _ = make_dataloaders(samples, _make_samples(4), batch_size=8)
        x, y = next(iter(dl))
        assert y.shape == (8,)
        assert y.dtype == torch.int64

    def test_batch_size_respected(self):
        samples = _make_samples(20)
        dl, _ = make_dataloaders(samples, _make_samples(4), batch_size=5)
        x, _ = next(iter(dl))
        assert x.shape[0] == 5

    def test_val_dataloader_not_empty(self):
        val_samples = _make_samples(6)
        _, val_dl = make_dataloaders(_make_samples(10), val_samples, batch_size=6)
        x, y = next(iter(val_dl))
        assert x.shape[0] == 6

    def test_labels_match_samples(self):
        samples = [{"image": [[0.0] * 14 for _ in range(14)], "label": c} for c in range(5)]
        dl, _ = make_dataloaders(samples, samples[:2], batch_size=5)
        _, y = next(iter(dl))
        assert set(y.tolist()) == set(range(5))


# ── export_onnx ───────────────────────────────────────────────────────────────


class TestExportOnnx:
    def test_creates_correct_subdirectories(self, small_model, tmp_path):
        export_dir = str(tmp_path)
        cls_dir, emb_dir = export_onnx(small_model, export_dir)
        assert os.path.isdir(cls_dir)
        assert os.path.isdir(emb_dir)
        assert cls_dir.endswith("model")
        assert emb_dir.endswith("embedder")

    def test_onnx_files_exist(self, small_model, tmp_path):
        cls_dir, emb_dir = export_onnx(small_model, str(tmp_path))
        assert os.path.isfile(os.path.join(cls_dir, "model.onnx"))
        assert os.path.isfile(os.path.join(emb_dir, "model.onnx"))

    def test_classifier_onnx_is_valid(self, small_model, tmp_path):
        cls_dir, _ = export_onnx(small_model, str(tmp_path))
        model_proto = onnx.load(os.path.join(cls_dir, "model.onnx"), load_external_data=True)
        onnx.checker.check_model(model_proto)  # raises if invalid

    def test_embedder_onnx_is_valid(self, small_model, tmp_path):
        _, emb_dir = export_onnx(small_model, str(tmp_path))
        model_proto = onnx.load(os.path.join(emb_dir, "model.onnx"), load_external_data=True)
        onnx.checker.check_model(model_proto)

    def test_classifier_onnx_runs(self, small_model, tmp_path):
        cls_dir, _ = export_onnx(small_model, str(tmp_path))
        sess = ort.InferenceSession(os.path.join(cls_dir, "model.onnx"))
        dummy = np.random.randn(2, INPUT_DIM).astype(np.float32)
        (logits,) = sess.run(["logits"], {"features": dummy})
        assert logits.shape == (2, NUM_CLASSES)

    def test_embedder_onnx_runs(self, small_model, tmp_path):
        _, emb_dir = export_onnx(small_model, str(tmp_path))
        sess = ort.InferenceSession(os.path.join(emb_dir, "model.onnx"))
        dummy = np.random.randn(2, INPUT_DIM).astype(np.float32)
        (embedding,) = sess.run(["embedding"], {"features": dummy})
        assert embedding.shape == (2, EMBEDDING_DIM)


# ── compute_reference_distributions ──────────────────────────────────────────


class TestComputeReferenceDistributions:
    def test_top_level_keys(self, small_model):
        samples = _make_samples(30)
        result = compute_reference_distributions(small_model, samples)
        assert "reference_distribution" in result
        assert "class_gaussians" in result

    def test_pixel_statistics_fields(self, small_model):
        samples = _make_samples(20)
        ref = compute_reference_distributions(small_model, samples)["reference_distribution"]
        assert "mean" in ref["pixel_statistics"]
        assert "std" in ref["pixel_statistics"]
        assert isinstance(ref["pixel_statistics"]["mean"], float)
        assert isinstance(ref["pixel_statistics"]["std"], float)

    def test_prediction_class_frequencies_sums_to_one(self, small_model):
        samples = _make_samples(30)
        ref = compute_reference_distributions(small_model, samples)["reference_distribution"]
        freqs = ref["prediction_class_frequencies"]
        assert len(freqs) == NUM_CLASSES
        assert abs(sum(freqs) - 1.0) < 1e-5

    def test_embedding_mean_length(self, small_model):
        samples = _make_samples(20)
        ref = compute_reference_distributions(small_model, samples)["reference_distribution"]
        assert len(ref["embedding_mean"]) == EMBEDDING_DIM

    def test_class_gaussians_only_for_present_classes(self, small_model):
        # Samples with only 2 distinct labels
        samples = [{"image": [[0.1] * 14 for _ in range(14)], "label": c} for c in [0, 1] * 10]
        gaussians = compute_reference_distributions(small_model, samples)["class_gaussians"][
            "classes"
        ]
        assert set(gaussians.keys()) == {"0", "1"}

    def test_class_gaussians_structure(self, small_model):
        samples = _make_samples(20)
        gaussians = compute_reference_distributions(small_model, samples)["class_gaussians"][
            "classes"
        ]
        for _cls_key, g in gaussians.items():
            assert "mean" in g
            assert "precision" in g
            assert "num_samples" in g
            assert len(g["mean"]) == EMBEDDING_DIM
            assert len(g["precision"]) == EMBEDDING_DIM
            assert len(g["precision"][0]) == EMBEDDING_DIM

    def test_num_samples_matches_input(self, small_model):
        n = 25
        samples = _make_samples(n)
        ref = compute_reference_distributions(small_model, samples)["reference_distribution"]
        assert ref["num_samples"] == n

    def test_single_sample_class_is_excluded_from_gaussians(self, small_model):
        # A class with exactly 1 sample would produce NaN via np.cov(ddof=1).
        # The fix: skip classes with < 2 samples.
        samples = (
            [{"image": [[0.1] * 14 for _ in range(14)], "label": 0}]  # 1 sample — skipped
            + [{"image": [[0.5] * 14 for _ in range(14)], "label": 1}] * 10  # 10 samples
        )
        gaussians = compute_reference_distributions(small_model, samples)["class_gaussians"][
            "classes"
        ]
        # class 0 must be absent — 1 sample is insufficient for a valid covariance
        assert "0" not in gaussians
        assert "1" in gaussians

    def test_no_nan_in_precision_matrices(self, small_model):
        # Guard against NaN leaking from np.cov when a class has very few samples.
        samples = _make_samples(20)
        gaussians = compute_reference_distributions(small_model, samples)["class_gaussians"][
            "classes"
        ]
        for cls_key, g in gaussians.items():
            for row in g["precision"]:
                for val in row:
                    assert not math.isnan(val), f"NaN in precision matrix for class {cls_key}"


# ── main() orchestration ──────────────────────────────────────────────────────


def _make_toy_samples(n: int) -> list[dict]:
    return [{"image": [[0.5] * 14 for _ in range(14)], "label": i % NUM_CLASSES} for i in range(n)]


class TestTrainingMain:
    def _make_mock_controller(self, run_id: str = "test-run-id") -> MagicMock:
        ctrl = MagicMock()

        @contextmanager
        def fake_start_run(experiment_name):
            yield run_id

        ctrl.start_run = fake_start_run
        ctrl.register_model.return_value = "1"
        return ctrl

    def test_raises_when_no_dataset_version(self):
        from training import main as training_main

        mock_dataset_ctrl = MagicMock()
        mock_dataset_ctrl.get_latest_version.return_value = None

        with (
            patch("training.main.DatasetController", return_value=mock_dataset_ctrl),
            patch("training.main.ModelArtifactController"),
            pytest.raises(RuntimeError, match="No dataset version found"),
        ):
            training_main.main()

    def test_raises_when_empty_train_split(self):
        from training import main as training_main

        mock_dataset_ctrl = MagicMock()
        mock_dataset_ctrl.get_latest_version.return_value = "v1"
        mock_dataset_ctrl.get_dataset_split.side_effect = lambda version, split: (
            [] if split == "train" else _make_toy_samples(5)
        )

        with (
            patch("training.main.DatasetController", return_value=mock_dataset_ctrl),
            patch("training.main.ModelArtifactController"),
            pytest.raises(RuntimeError, match="Dataset split is empty"),
        ):
            training_main.main()

    def test_raises_when_empty_val_split(self):
        from training import main as training_main

        mock_dataset_ctrl = MagicMock()
        mock_dataset_ctrl.get_latest_version.return_value = "v1"
        mock_dataset_ctrl.get_dataset_split.side_effect = lambda version, split: (
            [] if split == "val" else _make_toy_samples(5)
        )

        with (
            patch("training.main.DatasetController", return_value=mock_dataset_ctrl),
            patch("training.main.ModelArtifactController"),
            pytest.raises(RuntimeError, match="Dataset split is empty"),
        ):
            training_main.main()

    def test_full_pipeline_calls_all_controller_methods(self):
        from training import main as training_main

        train_samples = _make_toy_samples(20)
        val_samples = _make_toy_samples(10)

        mock_dataset_ctrl = MagicMock()
        mock_dataset_ctrl.get_latest_version.return_value = "v1"
        mock_dataset_ctrl.get_dataset_split.side_effect = lambda version, split: (
            train_samples if split == "train" else val_samples
        )

        mock_artifact_ctrl = self._make_mock_controller()

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {"val_loss": 0.5, "val_acc": 0.8, "val_f1": 0.79}

        with (
            patch("training.main.DatasetController", return_value=mock_dataset_ctrl),
            patch("training.main.ModelArtifactController", return_value=mock_artifact_ctrl),
            patch("training.main.pl.Trainer", return_value=mock_trainer),
        ):
            training_main.main()

        mock_artifact_ctrl.log_params.assert_called_once()
        mock_artifact_ctrl.log_metrics.assert_called_once()
        mock_artifact_ctrl.log_training_outputs.assert_called_once()
        mock_artifact_ctrl.register_model.assert_called_once()
        mock_artifact_ctrl.promote_model.assert_called_once()

    def test_full_pipeline_passes_correct_model_name(self):
        from training import main as training_main

        train_samples = _make_toy_samples(20)
        val_samples = _make_toy_samples(10)

        mock_dataset_ctrl = MagicMock()
        mock_dataset_ctrl.get_latest_version.return_value = "v1"
        mock_dataset_ctrl.get_dataset_split.side_effect = lambda version, split: (
            train_samples if split == "train" else val_samples
        )

        mock_artifact_ctrl = self._make_mock_controller()
        mock_artifact_ctrl.register_model.return_value = "42"

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {"val_loss": 0.5, "val_acc": 0.8, "val_f1": 0.79}

        with (
            patch("training.main.DatasetController", return_value=mock_dataset_ctrl),
            patch("training.main.ModelArtifactController", return_value=mock_artifact_ctrl),
            patch("training.main.pl.Trainer", return_value=mock_trainer),
        ):
            training_main.main()

        register_call = mock_artifact_ctrl.register_model.call_args
        model_name_arg = (
            register_call[0][1] if register_call[0] else register_call[1].get("model_name")
        )
        assert model_name_arg == "test-model"  # from conftest MODEL_NAME env var

        promote_call = mock_artifact_ctrl.promote_model.call_args
        assert "test-model" in promote_call[0]
