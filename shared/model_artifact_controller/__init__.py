# shared/model_artifact_controller/__init__.py
"""
Model artifact controller package — the single point of contact between the
rest of the system and the model storage backend (currently MLflow).

No other module should import mlflow directly. They should instantiate
MLflowModelArtifactController (or any future alternative) and call through
the ModelArtifactController Protocol.

Services import their specific implementation directly, e.g.:
  from shared.model_artifact_controller.mlflow import MLflowModelArtifactController
  from shared.model_artifact_controller import ModelArtifactError
"""

from shared.model_artifact_controller._protocol import ModelArtifactController, ModelArtifactError
from shared.model_artifact_controller.mlflow import MLflowModelArtifactController

__all__ = [
    "ModelArtifactError",
    "ModelArtifactController",
    "MLflowModelArtifactController",
]
