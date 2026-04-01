# shared/model_artifact_controller/_training_logger.py
"""TrainingLogger — high-level facade for producing model artifacts.

Used exclusively by the training pipeline.  Hides run IDs, artifact
paths, and backend details behind a single ``log()`` call that accepts
a strongly-typed ``TrainingArtifacts`` bundle.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from shared.config import require_env
from shared.model_artifact_controller._protocol import ModelArtifactError
from shared.model_artifact_controller.types import (
    ModelVersion,
    TrainingArtifacts,
)

if TYPE_CHECKING:
    from shared.model_artifact_controller._protocol import (
        ModelArtifactController as ModelArtifactControllerProtocol,
    )


class TrainingRun:
    """Opaque handle for an in-progress training run.

    Callers receive this from ``TrainingLogger.start()`` and pass it back
    to ``log()`` and ``register()``.  The underlying run identifier is
    intentionally hidden.

    Attributes:
        _run_id: Internal run identifier (not part of the public API).
    """

    __slots__ = ("_run_id",)

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id


class TrainingLogger:
    """Facade for logging training artifacts and registering models.

    Reads ``MODEL_NAME`` and backend configuration from environment
    variables.  No backend-specific types appear in the public API.

    Example::

        logger = TrainingLogger()
        with logger.start("experiment") as run:
            logger.log(run, artifacts)
            version = logger.register(run, promote=True)
    """

    def __init__(self) -> None:
        self._model_name = require_env("MODEL_NAME")
        backend_name = os.getenv("MODEL_ARTIFACT_BACKEND", "mlflow").strip().lower()
        if backend_name == "mlflow":
            from shared.model_artifact_controller.mlflow import (
                MLflowModelArtifactController,
            )

            self._backend: ModelArtifactControllerProtocol = MLflowModelArtifactController()
            return
        raise ModelArtifactError(
            f"Unsupported MODEL_ARTIFACT_BACKEND='{backend_name}'. Supported: mlflow"
        )

    @contextmanager
    def start(self, experiment_name: str) -> Generator[TrainingRun, None, None]:
        """Start a tracked training run and yield an opaque handle.

        Args:
            experiment_name: Logical name for the experiment group.

        Yields:
            A ``TrainingRun`` handle to pass to ``log()`` and ``register()``.

        Raises:
            ModelArtifactError: If the run cannot be started.
        """
        with self._backend.start_run(experiment_name) as run_id:
            yield TrainingRun(run_id)

    def log(self, run: TrainingRun, artifacts: TrainingArtifacts) -> None:
        """Log all training outputs in a single call.

        Records hyperparameters, evaluation metrics, the ONNX model
        directory, reference distribution, class Gaussians, and the
        feature schema.

        Args:
            run: Handle obtained from ``start()``.
            artifacts: Strongly-typed bundle of all training outputs.

        Raises:
            ModelArtifactError: If any artifact fails to upload.
        """
        run_id = run._run_id
        self._backend.log_params(run_id, {k: str(v) for k, v in artifacts.params.items()})
        self._backend.log_metrics(run_id, artifacts.metrics)
        self._backend.log_training_outputs(
            run_id=run_id,
            model_dir=artifacts.model_dir,
            reference_distribution=artifacts.reference_distribution.to_dict(),
            class_gaussians=artifacts.class_gaussians.to_dict(),
            feature_schema=artifacts.feature_schema.to_dict(),
        )

    def register(self, run: TrainingRun, *, promote: bool = False) -> ModelVersion:
        """Register the trained model and optionally promote it.

        Args:
            run: Handle obtained from ``start()``.
            promote: If ``True``, immediately set the new version as the
                Production model.

        Returns:
            An opaque ``ModelVersion`` handle.

        Raises:
            ModelArtifactError: If registration or promotion fails.
        """
        run_id = run._run_id
        version = self._backend.register_model(run_id, self._model_name)
        if promote:
            self._backend.promote_model(self._model_name, version)
        return ModelVersion(version=version, _model_name=self._model_name, _run_id=run_id)
