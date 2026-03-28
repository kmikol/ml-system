# shared/schemas/api.py
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image: list[list[float]] = Field(description="14x14 grayscale image, values in [0, 1]")
    uuid: UUID | None = Field(
        default=None,
        description="UUID of this sample, if known (enables annotation pipeline)",
    )


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str
    uuid: UUID


class ValidationErrorResponse(BaseModel):
    detail: str
    errors: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    uptime_seconds: float
