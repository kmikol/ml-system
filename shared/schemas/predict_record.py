# shared/schemas/predict_record.py
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from shared.schemas.feature_schema import EMBEDDING_DIM


class PredictRecord(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    model_version: str
    embedding: list[float] = Field(..., min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)
    prediction: int
    confidence: float
    prediction_distribution: list[float]
    mahalanobis_distance: float | None = None
    annotation_status: Literal["none", "candidate", "annotated"] = "none"
    annotated_label: int | None = None  # written by annotation job
