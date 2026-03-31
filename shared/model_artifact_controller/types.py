# shared/model_artifact_controller/types.py
"""Strongly-typed data structures for model artifact operations.

All types are frozen dataclasses with Google-style docstrings. They form
the public contract of the ``TrainingLogger`` and ``ModelStore`` facades
and must never expose backend-specific details (MLflow run IDs, artifact
paths, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelStage(str, Enum):
    """Deployment stage alias used by the model registry.

    Attributes:
        PRODUCTION: The live model serving real traffic.
        CANARY: A candidate model under evaluation.
    """

    PRODUCTION = "Production"
    CANARY = "Canary"


# ── Reference distribution ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReferenceDistribution:
    """Training-set statistics used for drift monitoring (PSI).

    Attributes:
        num_samples: Number of training samples the distribution was computed from.
        pixel_mean: Mean pixel intensity across the training set.
        pixel_std: Standard deviation of pixel intensity.
        embedding_mean: Per-dimension mean of the embedding vectors.
        embedding_cov: Full covariance matrix of the embedding vectors.
        prediction_class_frequencies: Normalised class frequency vector
            (length = ``num_classes``).
    """

    num_samples: int
    pixel_mean: float
    pixel_std: float
    embedding_mean: list[float]
    embedding_cov: list[list[float]]
    prediction_class_frequencies: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the canonical JSON-compatible dictionary."""
        return {
            "num_samples": self.num_samples,
            "pixel_statistics": {
                "mean": self.pixel_mean,
                "std": self.pixel_std,
            },
            "embedding_mean": self.embedding_mean,
            "embedding_cov": self.embedding_cov,
            "prediction_class_frequencies": self.prediction_class_frequencies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceDistribution:
        """Deserialise from the canonical JSON dictionary.

        Args:
            data: Dictionary with keys matching the training pipeline output.

        Returns:
            A new ``ReferenceDistribution`` instance.
        """
        ps = data.get("pixel_statistics", {})
        return cls(
            num_samples=data["num_samples"],
            pixel_mean=ps["mean"],
            pixel_std=ps["std"],
            embedding_mean=data["embedding_mean"],
            embedding_cov=data["embedding_cov"],
            prediction_class_frequencies=data["prediction_class_frequencies"],
        )


# ── Class Gaussians ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassGaussian:
    """Per-class Gaussian parameters for Mahalanobis distance scoring.

    Attributes:
        mean: Centroid of the class in embedding space.
        precision: Inverse covariance (precision) matrix.
        num_samples: Number of training samples used to fit this Gaussian.
    """

    mean: list[float]
    precision: list[list[float]]
    num_samples: int


@dataclass(frozen=True)
class ClassGaussians:
    """Collection of per-class Gaussian fits.

    Attributes:
        classes: Mapping from class label (as string) to its Gaussian fit.
    """

    classes: dict[str, ClassGaussian]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the canonical JSON-compatible dictionary."""
        return {
            "classes": {
                k: {
                    "mean": g.mean,
                    "precision": g.precision,
                    "num_samples": g.num_samples,
                }
                for k, g in self.classes.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassGaussians:
        """Deserialise from the canonical JSON dictionary.

        Args:
            data: Dictionary with a ``"classes"`` key mapping class labels
                to Gaussian parameter dicts.

        Returns:
            A new ``ClassGaussians`` instance.
        """
        return cls(
            classes={
                k: ClassGaussian(
                    mean=g["mean"],
                    precision=g["precision"],
                    num_samples=g["num_samples"],
                )
                for k, g in data["classes"].items()
            }
        )


# ── Feature schema ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureSchema:
    """Static schema describing model input/output dimensions.

    Attributes:
        image_size: Spatial dimensions of the input image (e.g. ``[14, 14]``).
        num_classes: Number of classification classes.
        input_dim: Flattened input dimension (``image_size[0] * image_size[1]``).
    """

    image_size: list[int]
    num_classes: int
    input_dim: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the canonical JSON-compatible dictionary."""
        return {
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
        }


# ── Training artifacts bundle ─────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingArtifacts:
    """Complete set of outputs from a training run.

    Passed as a single argument to ``TrainingLogger.log()`` so callers
    cannot forget to log a required artifact.

    Attributes:
        model_dir: Local path to the directory containing the exported ONNX model.
        params: Hyperparameters and metadata to record (values coerced to strings).
        metrics: Numeric evaluation metrics (e.g. ``val_loss``, ``val_acc``).
        reference_distribution: Training-set statistics for drift monitoring.
        class_gaussians: Per-class Gaussian fits for Mahalanobis scoring.
        feature_schema: Static input/output dimension metadata.
    """

    model_dir: str
    params: dict[str, str | int | float]
    metrics: dict[str, float]
    reference_distribution: ReferenceDistribution
    class_gaussians: ClassGaussians
    feature_schema: FeatureSchema


# ── Consumer-facing result types ──────────────────────────────────────────────


@dataclass(frozen=True)
class ServingBundle:
    """Downloaded model ready for inference.

    Attributes:
        model_path: Local filesystem path to the ONNX model file.
        class_gaussians: Per-class Gaussians, present only when
            ``include_gaussians=True`` was requested.
    """

    model_path: str
    class_gaussians: ClassGaussians | None = None


@dataclass(frozen=True)
class VersionArtifacts:
    """Artifacts for a specific model version (advanced API).

    Attributes:
        model_path: Local path to the ONNX model, or ``None`` if not requested.
        class_gaussians: Per-class Gaussians, or ``None`` if not requested.
        reference_distribution: Reference distribution, or ``None`` if not
            requested.
    """

    model_path: str | None = None
    class_gaussians: ClassGaussians | None = None
    reference_distribution: ReferenceDistribution | None = None


@dataclass(frozen=True)
class ModelVersion:
    """Opaque handle for a registered model version.

    Returned by ``TrainingLogger.register()`` and ``ModelStore.get_version_for_run()``.
    Can be passed to ``ModelStore.promote()`` or ``TrainingLogger.register(promote=True)``.

    Attributes:
        version: The version string assigned by the registry.
    """

    version: str
    _model_name: str = field(repr=False)
    _run_id: str = field(repr=False)
