# shared/model_artifact_controller/_protocol.py
"""ModelArtifactController protocol and shared error type."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


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

    def log_training_outputs(
        self,
        run_id: str,
        classifier_dir: str,
        embedder_dir: str,
        reference_distribution: dict[str, Any],
        class_gaussians: dict[str, Any],
        feature_schema: dict[str, Any],
    ) -> None:
        """Log canonical training outputs used by serving and drift callers."""
        ...

    def download_serving_bundle(self, run_id: str, local_dir: str) -> tuple[str, str, dict[str, Any] | None]:
        """Return classifier path, embedder path, and optional class Gaussians payload."""
        ...

    def download_reference_distribution(self, run_id: str, local_dir: str) -> dict[str, Any]:
        """Download and parse reference distribution payload for a run."""
        ...
