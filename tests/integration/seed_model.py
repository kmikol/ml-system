"""Seed a minimal ONNX model into MLflow for integration tests.

Creates a Classifier with random (untrained) weights, exports it to ONNX,
and registers + promotes it to the Production alias. No training data needed
— the graph only needs to be structurally valid for serving to load and run.

Usage (as a docker-compose one-shot service):
  docker-compose -f docker-compose.test.yml run model-seed

Environment:
  MODEL_NAME            - registered model name (e.g. ml_system_model)
  MLFLOW_TRACKING_URI   - MLflow backend URL
  MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from shared.config import require_env
from shared.model_artifact_controller.mlflow import MLflowModelArtifactController
from shared.schemas.feature_schema import EMBEDDING_DIM, INPUT_DIM, NUM_CLASSES
from training.model import Classifier, UnifiedWrapper


def export_minimal_onnx(model_dir: Path) -> None:
    """Export an untrained UnifiedWrapper to ONNX in model_dir/model.onnx."""
    clf = Classifier(
        input_dim=INPUT_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        lr=1e-3,
    )
    clf.eval()
    wrapper = UnifiedWrapper(clf)
    wrapper.eval()

    dummy = torch.zeros(1, INPUT_DIM)
    torch.onnx.export(
        wrapper,
        dummy,
        str(model_dir / "model.onnx"),
        input_names=["features"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "features": {0: "batch"},
            "logits": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )


def seed_one_model(model_name: str, controller: MLflowModelArtifactController) -> str:
    """Export a minimal ONNX model, register it in MLflow, and promote to Production.

    Returns the version string assigned by the registry.
    Called by the seed docker-compose service and also by the hot-swap test.
    """
    with tempfile.TemporaryDirectory(prefix="seed_model_") as tmp:
        model_dir = Path(tmp) / "model"
        model_dir.mkdir()
        export_minimal_onnx(model_dir)

        ref_dist = {
            "num_samples": 100,
            "pixel_statistics": {"mean": 0.1, "std": 0.1},
            "embedding_mean": [0.0] * EMBEDDING_DIM,
            "embedding_cov": [
                [1.0 if i == j else 0.0 for j in range(EMBEDDING_DIM)]
                for i in range(EMBEDDING_DIM)
            ],
            "prediction_class_frequencies": [1.0 / NUM_CLASSES] * NUM_CLASSES,
        }
        class_gaussians = {
            "classes": {
                str(i): {
                    "mean": [0.0] * EMBEDDING_DIM,
                    "precision": [
                        [1.0 if j == k else 0.0 for k in range(EMBEDDING_DIM)]
                        for j in range(EMBEDDING_DIM)
                    ],
                    "num_samples": 10,
                }
                for i in range(NUM_CLASSES)
            }
        }
        feature_schema = {
            "image_size": [14, 14],
            "num_classes": NUM_CLASSES,
            "input_dim": INPUT_DIM,
        }

        with controller.start_run("ci-seed") as run_id:
            controller.log_training_outputs(
                run_id,
                str(model_dir),
                ref_dist,
                class_gaussians,
                feature_schema,
            )
            version = controller.register_model(run_id, model_name)

        controller.promote_model(model_name, version)
        return version


def main() -> None:
    model_name = require_env("MODEL_NAME")
    controller = MLflowModelArtifactController()
    version = seed_one_model(model_name, controller)
    print(f"Seeded '{model_name}' version {version} → Production")


if __name__ == "__main__":
    main()
