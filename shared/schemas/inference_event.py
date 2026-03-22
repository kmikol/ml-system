# shared/schemas/inference_event.py
from datetime import datetime

from pydantic import BaseModel, Field


class InferenceEvent(BaseModel):
    event_id: str
    timestamp: datetime
    model_version: str
    request_id: str
    features: dict[str, float]
    embedding: list[float]
    prediction: int
    confidence: float
    prediction_distribution: list[float]
    schema_version: str = Field(default="1.0")
