# shared/model_artifact_controller/mlflow.py
"""MLflowModelArtifactController — MLflow backend for model artifact storage."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from shared.config import require_env
from shared.model_artifact_controller._protocol import ModelArtifactError

_ONNX_ROOT = "onnx"
_MODEL_DIR = "model"
_ONNX_FILENAME = "model.onnx"
_MLFLOW_PATH_MODEL = f"{_ONNX_ROOT}/{_MODEL_DIR}"
_REFERENCE_DIST_FILENAME = "reference_distribution.json"
_CLASS_GAUSSIANS_FILENAME = "class_gaussians.json"
_FEATURE_SCHEMA_FILENAME = "feature_schema.json"


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

    def log_training_outputs(
        self,
        run_id: str,
        model_dir: str,
        reference_distribution: dict[str, Any],
        class_gaussians: dict[str, Any],
        feature_schema: dict[str, Any],
    ) -> None:
        """Log canonical training artifacts without leaking path contracts to callers."""
        self.log_artifacts(run_id, model_dir, _MLFLOW_PATH_MODEL)

        with tempfile.TemporaryDirectory(prefix="mlflow_meta_") as tmpdir:
            ref_path = os.path.join(tmpdir, _REFERENCE_DIST_FILENAME)
            with open(ref_path, "w") as f:
                json.dump(reference_distribution, f)
            self.log_artifact(run_id, ref_path)

            gauss_path = os.path.join(tmpdir, _CLASS_GAUSSIANS_FILENAME)
            with open(gauss_path, "w") as f:
                json.dump(class_gaussians, f)
            self.log_artifact(run_id, gauss_path)

            schema_path = os.path.join(tmpdir, _FEATURE_SCHEMA_FILENAME)
            with open(schema_path, "w") as f:
                json.dump(feature_schema, f, indent=2)
            self.log_artifact(run_id, schema_path)

    def download_serving_bundle(
        self, run_id: str, local_dir: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Return local model path and optional class Gaussians payload."""
        onnx_dir = self.download_artifacts(run_id, _ONNX_ROOT, local_dir)
        model_path = self._resolve_onnx_path(onnx_dir, _MODEL_DIR)

        class_gaussians = None
        try:
            gauss_path = self.download_artifacts(run_id, _CLASS_GAUSSIANS_FILENAME, local_dir)
            with open(gauss_path) as f:
                class_gaussians = json.load(f)
        except Exception:
            class_gaussians = None

        return model_path, class_gaussians

    def download_reference_distribution(self, run_id: str, local_dir: str) -> dict[str, Any]:
        """Download and parse the canonical reference distribution JSON for a run."""
        try:
            path = self.download_artifacts(run_id, _REFERENCE_DIST_FILENAME, local_dir)
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to load reference distribution from run '{run_id}': {exc}"
            ) from exc

    def _resolve_onnx_path(self, onnx_download_dir: str, subdir: str) -> str:
        path = os.path.join(onnx_download_dir, subdir, _ONNX_FILENAME)
        if not os.path.isfile(path):
            self._dump_tree(onnx_download_dir, f"{subdir}/{_ONNX_FILENAME}")
        return path

    def _dump_tree(self, root: str, expected: str) -> None:
        import logging

        logger = logging.getLogger(__name__)
        logger.error("Expected file '%s' not found in '%s'", expected, root)
        logger.error("Actual contents:")
        for dirpath, _dirnames, filenames in os.walk(root):
            level = dirpath.replace(root, "").count(os.sep)
            indent = "  " * level
            logger.error("%s%s/", indent, os.path.basename(dirpath))
            for filename in filenames:
                fpath = os.path.join(dirpath, filename)
                size = os.path.getsize(fpath)
                logger.error("%s  %s  (%d bytes)", indent, filename, size)
        raise FileNotFoundError(f"'{expected}' not found in '{root}'")

    def register_model(self, run_id: str, model_name: str) -> str:
        """Register the unified model ONNX artifact from *run_id* under *model_name*.

        The registered model URI points to the ``onnx/model`` subdirectory
        of the run's artifacts. Returns the version string assigned by MLflow.
        """
        try:
            uri = f"runs:/{run_id}/{_MLFLOW_PATH_MODEL}"
            result = self._mlflow.register_model(uri, model_name)
            return str(result.version)
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to register model '{model_name}' from run '{run_id}': {exc}"
            ) from exc

    def promote_model(self, model_name: str, version: str, alias: str = "Production") -> None:
        """Set *alias* on *version* of *model_name*.

        Uses the MLflow aliases API (introduced in 2.9.0) instead of the
        deprecated stages API. Only one version can hold a given alias at a
        time — MLflow moves it automatically.
        """
        try:
            self._client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version,
            )
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to set alias '{alias}' on '{model_name}' v{version}: {exc}"
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

    def get_run_metrics(self, run_id: str) -> dict[str, float]:
        """Return all numeric metrics logged to *run_id*.

        Args:
            run_id: The run identifier.

        Returns:
            Dictionary mapping metric names to their float values.

        Raises:
            ModelArtifactError: If the run cannot be retrieved.
        """
        try:
            run = self._client.get_run(run_id)
            return {k: float(v) for k, v in run.data.metrics.items()}
        except Exception as exc:
            raise ModelArtifactError(f"Failed to get metrics for run '{run_id}': {exc}") from exc

    def search_version_by_run(self, model_name: str, run_id: str) -> str | None:
        """Return the version string registered from *run_id*, or ``None``.

        Args:
            model_name: Registered model name.
            run_id: The run identifier to search for.

        Returns:
            Version string if found, ``None`` otherwise.

        Raises:
            ModelArtifactError: If the search itself fails.
        """
        try:
            versions = self._client.search_model_versions(
                f"name='{model_name}' and run_id='{run_id}'"
            )
            return versions[0].version if versions else None
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to search versions for run '{run_id}': {exc}"
            ) from exc
