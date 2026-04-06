# shared/model_artifact_controller/_model_store.py
"""ModelStore — high-level facade for consuming model artifacts.

Used by serving, monitoring, and evaluation.  Hides run IDs and
artifact paths behind stage-based lookups that return strongly-typed
results.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from shared.config import require_env
from shared.model_artifact_controller._protocol import ModelArtifactError
from shared.model_artifact_controller.types import (
    ClassGaussians,
    ModelStage,
    ModelVersion,
    ReferenceDistribution,
    ServingBundle,
    VersionArtifacts,
)

if TYPE_CHECKING:
    from shared.model_artifact_controller._protocol import (
        ModelArtifactController as ModelArtifactControllerProtocol,
    )


class ModelStore:
    """Facade for downloading model artifacts by deployment stage.

    Reads ``MODEL_NAME`` and backend configuration from environment
    variables.  Callers never need a run ID — they request artifacts
    by ``ModelStage`` (Production, Canary, etc.).

    Example::

        store = ModelStore()
        bundle = store.get_serving_bundle(ModelStage.PRODUCTION, include_gaussians=True)
        session = ort.InferenceSession(bundle.model_path)
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

    # ── Default (simple) API ──────────────────────────────────────────────────

    def get_serving_bundle(
        self,
        stage: ModelStage = ModelStage.PRODUCTION,
        *,
        include_gaussians: bool = False,
    ) -> ServingBundle:
        """Download the model for inference at the given stage.

        Resolves the stage to a specific model version internally and
        downloads the ONNX model file.  Class Gaussians for Mahalanobis
        scoring are only fetched when explicitly requested.

        Args:
            stage: Deployment stage to fetch (default ``PRODUCTION``).
            include_gaussians: If ``True``, also download per-class Gaussian
                parameters.  Defaults to ``False`` because most callers
                only need the model itself.

        Returns:
            A ``ServingBundle`` with the local model path and optional
            class Gaussians.

        Raises:
            ModelArtifactError: If the stage has no model or download fails.
        """
        run_id = self._backend.get_production_run_id(self._model_name, stage.value)
        local_dir = tempfile.mkdtemp(prefix="ml_model_")
        model_path, raw_gaussians = self._backend.download_serving_bundle(run_id, local_dir)
        gaussians = None
        if include_gaussians and raw_gaussians is not None:
            try:
                gaussians = ClassGaussians.from_dict(raw_gaussians)
            except Exception as exc:
                raise ModelArtifactError(
                    f"Invalid class_gaussians payload from run '{run_id}': {exc}"
                ) from exc
        return ServingBundle(model_path=model_path, class_gaussians=gaussians)

    def get_reference_distribution(
        self,
        stage: ModelStage = ModelStage.PRODUCTION,
    ) -> ReferenceDistribution:
        """Download the reference distribution for drift monitoring.

        Args:
            stage: Deployment stage to fetch (default ``PRODUCTION``).

        Returns:
            A strongly-typed ``ReferenceDistribution``.

        Raises:
            ModelArtifactError: If the stage has no model or download fails.
        """
        run_id = self._backend.get_production_run_id(self._model_name, stage.value)
        local_dir = tempfile.mkdtemp(prefix="ml_ref_")
        raw = self._backend.download_reference_distribution(run_id, local_dir)
        return ReferenceDistribution.from_dict(raw)

    def get_current_version_id(
        self,
        stage: ModelStage = ModelStage.PRODUCTION,
    ) -> str:
        """Return an opaque identifier for the current model at *stage*.

        Use this for change-detection polling: compare the returned value
        with a cached copy to decide whether to re-download.  The
        identifier is intentionally opaque — do not parse or interpret it.

        Args:
            stage: Deployment stage to query (default ``PRODUCTION``).

        Returns:
            An opaque version identifier string.

        Raises:
            ModelArtifactError: If no model exists at the given stage.
        """
        return self._backend.get_production_run_id(self._model_name, stage.value)

    def get_metrics(
        self,
        stage: ModelStage = ModelStage.PRODUCTION,
    ) -> dict[str, float]:
        """Retrieve logged metrics for the current model at *stage*.

        Args:
            stage: Deployment stage to query (default ``PRODUCTION``).

        Returns:
            Dictionary mapping metric names to float values
            (e.g. ``{"val_acc": 0.97, "val_loss": 0.08}``).

        Raises:
            ModelArtifactError: If no model exists at the given stage or
                metrics cannot be retrieved.
        """
        run_id = self._backend.get_production_run_id(self._model_name, stage.value)
        return self._backend.get_run_metrics(run_id)

    # ── Advanced API (by version ID) ──────────────────────────────────────────

    def get_metrics_by_run(self, run_id: str) -> dict[str, float]:
        """Retrieve logged metrics for a specific run (advanced).

        Unlike ``get_metrics()``, this accepts a raw run identifier
        rather than resolving by stage.  Use this when you need metrics
        from a run that is not yet deployed (e.g. a candidate under
        evaluation).

        Args:
            run_id: The training run identifier.

        Returns:
            Dictionary mapping metric names to float values.

        Raises:
            ModelArtifactError: If the run cannot be retrieved.
        """
        return self._backend.get_run_metrics(run_id)

    def get_version_artifacts(
        self,
        version_id: str,
        *,
        include_gaussians: bool = False,
        include_reference: bool = False,
    ) -> VersionArtifacts:
        """Download artifacts for a specific model version.

        This is the advanced API for cases where you need artifacts from
        a version that is not the current stage model — for example,
        comparing multiple model versions in a monitoring dashboard.

        Args:
            version_id: Opaque version identifier (as returned by
                ``get_current_version_id()``).
            include_gaussians: Fetch per-class Gaussian parameters.
            include_reference: Fetch the reference distribution.

        Returns:
            A ``VersionArtifacts`` with the requested data populated.

        Raises:
            ModelArtifactError: If download or parsing fails.
        """
        local_dir = tempfile.mkdtemp(prefix="ml_version_")
        model_path, raw_gaussians = self._backend.download_serving_bundle(version_id, local_dir)

        gaussians = None
        if include_gaussians and raw_gaussians is not None:
            try:
                gaussians = ClassGaussians.from_dict(raw_gaussians)
            except Exception as exc:
                raise ModelArtifactError(f"Invalid class_gaussians payload: {exc}") from exc

        ref = None
        if include_reference:
            ref_dir = tempfile.mkdtemp(prefix="ml_version_ref_")
            raw_ref = self._backend.download_reference_distribution(version_id, ref_dir)
            ref = ReferenceDistribution.from_dict(raw_ref)

        return VersionArtifacts(
            model_path=model_path,
            class_gaussians=gaussians,
            reference_distribution=ref,
        )

    def get_version_for_run(self, run_id: str) -> ModelVersion | None:
        """Find the registered model version created by a specific run.

        This is primarily used by the evaluate-and-promote script to
        locate the version that should be promoted.

        Args:
            run_id: The training run identifier.

        Returns:
            A ``ModelVersion`` handle, or ``None`` if no version was
            registered from that run.

        Raises:
            ModelArtifactError: If the search itself fails.
        """
        version = self._backend.search_version_by_run(self._model_name, run_id)
        if version is None:
            return None
        return ModelVersion(version=version, _model_name=self._model_name, _run_id=run_id)

    def promote(self, version: ModelVersion, stage: ModelStage = ModelStage.PRODUCTION) -> None:
        """Promote a model version to the given stage alias.

        Args:
            version: Handle obtained from ``get_version_for_run()``.
            stage: Target stage alias (default ``PRODUCTION``).

        Raises:
            ModelArtifactError: If promotion fails.
        """
        self._backend.promote_model(version._model_name, version.version, alias=stage.value)

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        """Log a single metric to a run (for evaluation baselines).

        Args:
            run_id: The run to log to.
            key: Metric name.
            value: Metric value.

        Raises:
            ModelArtifactError: If logging fails.
        """
        self._backend.log_metrics(run_id, {key: value})
