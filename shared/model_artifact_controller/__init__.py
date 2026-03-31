# shared/model_artifact_controller/__init__.py
"""
Model artifact controller package — the single point of contact between the
rest of the system and the model storage backend (currently MLflow).

Two facades are provided, split by role:

- ``TrainingLogger``: used by the training pipeline to log artifacts and
  register models.
- ``ModelStore``: used by serving, monitoring, and evaluation to download
  artifacts by deployment stage.

No service should import backend-specific controllers directly.

Example (training)::

    from shared.model_artifact_controller import TrainingLogger, TrainingArtifacts
    logger = TrainingLogger()
    with logger.start("experiment") as run:
        logger.log(run, artifacts)
        logger.register(run, promote=True)

Example (serving)::

    from shared.model_artifact_controller import ModelStore, ModelStage
    store = ModelStore()
    bundle = store.get_serving_bundle(ModelStage.PRODUCTION, include_gaussians=True)
"""

from shared.model_artifact_controller._model_store import ModelStore
from shared.model_artifact_controller._protocol import ModelArtifactError
from shared.model_artifact_controller._training_logger import TrainingLogger, TrainingRun
from shared.model_artifact_controller.types import (
    ClassGaussian,
    ClassGaussians,
    FeatureSchema,
    ModelStage,
    ModelVersion,
    ReferenceDistribution,
    ServingBundle,
    TrainingArtifacts,
    VersionArtifacts,
)

__all__ = [
    "ClassGaussian",
    "ClassGaussians",
    "FeatureSchema",
    "ModelArtifactError",
    "ModelStage",
    "ModelStore",
    "ModelVersion",
    "ReferenceDistribution",
    "ServingBundle",
    "TrainingArtifacts",
    "TrainingLogger",
    "TrainingRun",
    "VersionArtifacts",
]
