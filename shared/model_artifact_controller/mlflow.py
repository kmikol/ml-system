# shared/model_artifact_controller/mlflow.py
"""MLflowModelArtifactController — MLflow backend for model artifact storage."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from shared.artifact_paths import MLFLOW_PATH_CLASSIFIER
from shared.config import require_env
from shared.model_artifact_controller._protocol import ModelArtifactError


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
        """Set the ``Production`` alias on *version* of *model_name*.

        Uses the MLflow aliases API (introduced in 2.9.0) instead of the
        deprecated stages API. Only one version can hold the ``Production``
        alias at a time — MLflow moves it automatically.
        """
        try:
            self._client.set_registered_model_alias(
                name=model_name,
                alias="Production",
                version=version,
            )
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to promote '{model_name}' v{version} to Production: {exc}"
            ) from exc

    def get_production_run_id(self, model_name: str, stage: str) -> str:
        """Return the ``run_id`` for the version of *model_name* with alias *stage*.

        Typical usage: ``stage="Production"``. Raises ``ModelArtifactError``
        if no version of *model_name* carries that alias.
        """
        try:
            mv = self._client.get_model_version_by_alias(name=model_name, alias=stage)
            return mv.run_id
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
