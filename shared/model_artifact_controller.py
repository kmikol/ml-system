# shared/model_artifact_controller.py
"""
Model artifact controller — the single point of contact between the rest of
the system and the model storage backend (currently MLflow).

No other module should import mlflow directly. They should instantiate
MLflowModelArtifactController (or any future alternative) and call through
the ModelArtifactController Protocol.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

from shared.artifact_paths import MLFLOW_PATH_CLASSIFIER
from shared.config import require_env


class ModelArtifactError(Exception):
    """Raised when any model artifact operation fails.

    No mlflow exception types escape this module — all failures are wrapped
    in this single type so callers write ``except ModelArtifactError`` instead
    of importing mlflow error classes.
    """


class ModelArtifactController(Protocol):
    """Interface for model artifact storage operations.

    Write new backend implementations as classes that satisfy this Protocol.
    Nothing else in the codebase needs to change.
    """

    def start_run(self, experiment_name: str) -> AbstractContextManager[str]:
        """Context manager that starts a tracked run and yields its run_id."""
        ...

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Log hyperparameters or other key/value metadata to a run."""
        ...

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """Log numeric evaluation metrics to a run."""
        ...

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        """Upload a single file to a run's artifact store."""
        ...

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str | None = None) -> None:
        """Upload all files in a directory to a run's artifact store."""
        ...

    def register_model(self, run_id: str, model_name: str) -> str:
        """Register artifacts from *run_id* under *model_name* and return the version string."""
        ...

    def promote_model(self, model_name: str, version: str) -> None:
        """Mark *version* of *model_name* as the production model."""
        ...

    def get_production_run_id(self, model_name: str, stage: str) -> str:
        """Return the run_id for the current *stage* version of *model_name*.

        Raises ModelArtifactError if no version is in that stage.
        """
        ...

    def download_artifacts(self, run_id: str, artifact_path: str, local_dir: str) -> str:
        """Download the artifact subtree at *artifact_path* to *local_dir*.

        Returns the path to the downloaded root directory.
        """
        ...


class MLflowModelArtifactController:
    """MLflow implementation of ModelArtifactController.

    Reads MLFLOW_TRACKING_URI from the environment on construction (crashes if
    missing). All mlflow imports are deferred to this class so services that do
    not construct it don't need mlflow installed.
    """

    def __init__(self) -> None:
        """Connect to MLflow.

        Reads ``MLFLOW_TRACKING_URI`` from the environment and crashes immediately
        if it is not set. MLflow is imported lazily here so services that never
        instantiate this class do not need mlflow installed.
        """
        self._tracking_uri = require_env("MLFLOW_TRACKING_URI")
        import mlflow  # lazy — keeps mlflow out of import-time for non-users

        self._mlflow = mlflow
        self._mlflow.set_tracking_uri(self._tracking_uri)
        self._client = mlflow.tracking.MlflowClient()

    @contextmanager
    def start_run(self, experiment_name: str) -> Generator[str, None, None]:
        """Start an MLflow run under *experiment_name* and yield its ``run_id``.

        Creates the experiment if it does not already exist. All exceptions
        except re-raised ``ModelArtifactError`` are wrapped in
        ``ModelArtifactError``.
        """
        try:
            self._mlflow.set_experiment(experiment_name)
            with self._mlflow.start_run() as run:
                yield run.info.run_id
        except ModelArtifactError:
            raise
        except Exception as exc:
            raise ModelArtifactError(
                f"Training run in experiment '{experiment_name}' failed: {exc}"
            ) from exc

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Log key/value hyperparameters to *run_id*. Values are coerced to strings."""
        try:
            for key, value in params.items():
                self._client.log_param(run_id, str(key), str(value))
        except Exception as exc:
            raise ModelArtifactError(f"Failed to log params to run '{run_id}': {exc}") from exc

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """Log numeric evaluation metrics to *run_id*. Values are coerced to float."""
        try:
            for key, value in metrics.items():
                self._client.log_metric(run_id, str(key), float(value))
        except Exception as exc:
            raise ModelArtifactError(f"Failed to log metrics to run '{run_id}': {exc}") from exc

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        """Upload a single file at *local_path* to *run_id*'s artifact store.

        *artifact_path* sets the subdirectory within the artifact store;
        defaults to the run root if ``None``.
        """
        try:
            self._client.log_artifact(run_id, local_path, artifact_path)
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to log artifact '{local_path}' to run '{run_id}': {exc}"
            ) from exc

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str | None = None) -> None:
        """Upload all files in *local_dir* to *run_id*'s artifact store.

        *artifact_path* sets the target subdirectory within the artifact store;
        defaults to the run root if ``None``.
        """
        try:
            self._client.log_artifacts(run_id, local_dir, artifact_path)
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to log artifacts from '{local_dir}' to run '{run_id}': {exc}"
            ) from exc

    def register_model(self, run_id: str, model_name: str) -> str:
        """Register the classifier ONNX artifact from *run_id* under *model_name*.

        The registered model URI points to the ``onnx/classifier`` subdirectory
        of the run's artifacts — the canonical artifact path defined in
        ``shared.artifact_paths``. Returns the version string assigned by MLflow.
        """
        try:
            uri = f"runs:/{run_id}/{MLFLOW_PATH_CLASSIFIER}"
            result = self._mlflow.register_model(uri, model_name)
            return str(result.version)
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to register model '{model_name}' from run '{run_id}': {exc}"
            ) from exc

    def promote_model(self, model_name: str, version: str) -> None:
        """Transition *version* of *model_name* to the ``Production`` stage.

        All other versions of *model_name* are archived automatically
        (``archive_existing_versions=True``).
        """
        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to promote '{model_name}' v{version} to Production: {exc}"
            ) from exc

    def get_production_run_id(self, model_name: str, stage: str) -> str:
        """Return the ``run_id`` for the version of *model_name* currently in *stage*.

        Typical usage: ``stage="Production"``. Raises ``ModelArtifactError``
        if no version of *model_name* is in that stage.
        """
        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")
            prod = next((v for v in versions if v.current_stage == stage), None)
            if prod is None:
                raise ModelArtifactError(f"No model '{model_name}' in stage '{stage}'")
            return prod.run_id
        except ModelArtifactError:
            raise
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to find '{stage}' version of '{model_name}': {exc}"
            ) from exc

    def download_artifacts(self, run_id: str, artifact_path: str, local_dir: str) -> str:
        """Download the artifact subtree at *artifact_path* from *run_id* to *local_dir*.

        Returns the local path of the downloaded root directory. Delegates
        directly to ``MlflowClient.download_artifacts``.
        """
        try:
            return self._client.download_artifacts(run_id, artifact_path, local_dir)
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to download '{artifact_path}' from run '{run_id}': {exc}"
            ) from exc
