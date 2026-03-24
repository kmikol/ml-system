# shared/model_artifact_controller/__init__.py
"""
Model artifact controller package — the single point of contact between the
rest of the system and the model storage backend (currently MLflow).

No service should import backend-specific controllers directly. Services should
instantiate ModelArtifactController, which selects the configured backend and
exposes one stable interface.

Example:
  from shared.model_artifact_controller import ModelArtifactController
  from shared.model_artifact_controller import ModelArtifactError
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Any

from shared.model_artifact_controller._protocol import (
    ModelArtifactController as ModelArtifactControllerProtocol,
)
from shared.model_artifact_controller._protocol import ModelArtifactError
from shared.model_artifact_controller.mlflow import MLflowModelArtifactController


class ModelArtifactController:
    """Facade over concrete model artifact backends.

    Backend selection is environment-driven via MODEL_ARTIFACT_BACKEND.
    Supported values:
      - "mlflow" (default)
    """

    def __init__(self) -> None:
        backend_name = os.getenv("MODEL_ARTIFACT_BACKEND", "mlflow").strip().lower()
        if backend_name == "mlflow":
            self._backend: ModelArtifactControllerProtocol = MLflowModelArtifactController()
            return

        raise ModelArtifactError(
            f"Unsupported MODEL_ARTIFACT_BACKEND='{backend_name}'. "
            "Supported backends: mlflow"
        )

    def start_run(self, experiment_name: str) -> AbstractContextManager[str]:
        return self._backend.start_run(experiment_name)

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        self._backend.log_params(run_id, params)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self._backend.log_metrics(run_id, metrics)

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        self._backend.log_artifact(run_id, local_path, artifact_path)

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str | None = None) -> None:
        self._backend.log_artifacts(run_id, local_dir, artifact_path)

    def log_training_outputs(
        self,
        run_id: str,
        classifier_dir: str,
        embedder_dir: str,
        reference_distribution: dict[str, Any],
        class_gaussians: dict[str, Any],
        feature_schema: dict[str, Any],
    ) -> None:
        self._backend.log_training_outputs(
            run_id=run_id,
            classifier_dir=classifier_dir,
            embedder_dir=embedder_dir,
            reference_distribution=reference_distribution,
            class_gaussians=class_gaussians,
            feature_schema=feature_schema,
        )

    def register_model(self, run_id: str, model_name: str) -> str:
        return self._backend.register_model(run_id, model_name)

    def promote_model(self, model_name: str, version: str) -> None:
        self._backend.promote_model(model_name, version)

    def get_production_run_id(self, model_name: str, stage: str) -> str:
        return self._backend.get_production_run_id(model_name, stage)

    def download_artifacts(self, run_id: str, artifact_path: str, local_dir: str) -> str:
        return self._backend.download_artifacts(run_id, artifact_path, local_dir)

    def download_serving_bundle(self, run_id: str, local_dir: str) -> tuple[str, str, dict[str, Any] | None]:
        return self._backend.download_serving_bundle(run_id, local_dir)

    def download_reference_distribution(self, run_id: str, local_dir: str) -> dict[str, Any]:
        return self._backend.download_reference_distribution(run_id, local_dir)

__all__ = [
    "ModelArtifactError",
    "ModelArtifactControllerProtocol",
    "ModelArtifactController",
    "MLflowModelArtifactController",
]
