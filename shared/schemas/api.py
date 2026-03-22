# shared/schemas/api.py
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image: list[list[float]] = Field(description="14x14 grayscale image, values in [0, 1]")
    request_id: Optional[str] = Field(default=None, description="Optional trace ID")


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str
    request_id: str


class ValidationErrorResponse(BaseModel):
    detail: str
    errors: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float
