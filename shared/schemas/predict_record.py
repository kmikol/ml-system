# shared/schemas/predict_record.py
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictRecord(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    model_version: str
    image: list[list[float]]
    embedding: list[float]
    prediction: int
    confidence: float
    prediction_distribution: list[float]
    label: Optional[int] = None
    annotation_status: Literal["none", "candidate", "annotated"] = "none"
