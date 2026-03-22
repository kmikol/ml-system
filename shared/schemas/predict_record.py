# shared/schemas/predict_record.py
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictRecord(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    model_version: str
    features: dict[str, float]
    embedding: list[float]
    prediction: int
    confidence: float
    prediction_distribution: list[float]
    label: int | None = None
    annotation_status: Literal["none", "candidate", "annotated"] = "none"
