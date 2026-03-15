# shared/schemas/inference_event.py
from pydantic import BaseModel, Field
from datetime import datetime


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
