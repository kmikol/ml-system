# shared/schemas/api.py
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request payload for `/predict` inference calls.

    Carries one preprocessed grayscale image and optional UUID used by the
    annotation feedback loop.
    """

    image: list[list[float]] = Field(description="14x14 grayscale image, values in [0, 1]")
    uuid: UUID | None = Field(
        default=None,
        description="UUID of this sample, if known (enables annotation pipeline)",
    )


class PredictResponse(BaseModel):
    """Successful response payload returned by `/predict`."""

    prediction: int = Field(description="Predicted class index.")
    confidence: float = Field(description="Confidence score for predicted class.")
    model_version: str = Field(description="Model version used to generate prediction.")
    uuid: UUID = Field(description="UUID associated with this prediction record.")


class ValidationErrorResponse(BaseModel):
    """Validation error payload returned when request image is invalid."""

    detail: str = Field(description="High-level error summary.")
    errors: list[dict] = Field(description="Detailed validation errors.")


class HealthResponse(BaseModel):
    """Health endpoint payload for serving liveness/readiness checks."""

    status: str = Field(description="Service health state, e.g. healthy/unhealthy.")
    model_loaded: bool = Field(description="Whether model is currently loaded in memory.")
    model_version: str | None = Field(
        default=None,
        description="Loaded model version, if available.",
    )
    uptime_seconds: float = Field(description="Process uptime in seconds.")
