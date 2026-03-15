# shared/schemas/api.py
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(description="Input features keyed by name")
    request_id: str | None = Field(default=None, description="Optional trace ID")


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
    model_version: str | None = None
    uptime_seconds: float
