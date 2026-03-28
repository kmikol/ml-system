# shared/schemas/predict_record.py
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PredictRecord(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    model_version: str
    embedding: list[float]
    prediction: int
    confidence: float
    prediction_distribution: list[float]
    annotation_status: Literal["none", "candidate", "annotated"] = "none"
    annotated_label: Optional[int] = None  # written by annotation job
