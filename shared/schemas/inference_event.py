# shared/schemas/inference_event.py
from datetime import datetime

from pydantic import BaseModel, Field


class InferenceEvent(BaseModel):
    """Event envelope for emitted inference payloads.

    Used when serializing prediction outputs for downstream processing and
    observability pipelines.
    """

    event_id: str = Field(description="Unique event identifier.")
    timestamp: datetime = Field(description="UTC timestamp of inference event.")
    model_version: str = Field(description="Model version used for this inference.")
    request_id: str = Field(description="Request correlation identifier.")
    image: list[list[float]] = Field(description="Input 14x14 grayscale image.")
    embedding: list[float] = Field(description="Model embedding vector for the sample.")
    prediction: int = Field(description="Predicted class index.")
    confidence: float = Field(description="Confidence score for predicted class.")
    prediction_distribution: list[float] = Field(
        description="Full class-probability distribution.",
    )
    schema_version: str = Field(default="1.0", description="Event payload schema version.")
