# shared/schemas/predict_record.py
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from shared.schemas.feature_schema import EMBEDDING_DIM


class PredictRecord(BaseModel):
    """Stored inference record persisted by serving and reused by feedback loops.

    This schema is the canonical prediction row representation used by data
    controller operations and downstream monitoring/annotation pipelines.
    """

    uuid: UUID = Field(default_factory=uuid4, description="Unique prediction record identifier.")
    timestamp: datetime = Field(description="UTC timestamp when prediction was produced.")
    model_version: str = Field(description="Model version that generated this prediction.")
    embedding: list[float] = Field(
        ...,
        min_length=EMBEDDING_DIM,
        max_length=EMBEDDING_DIM,
        description="Embedding vector extracted from serving model.",
    )
    prediction: int = Field(description="Predicted class index.")
    confidence: float = Field(description="Confidence score of predicted class.")
    prediction_distribution: list[float] = Field(
        description="Full class-probability distribution returned by model.",
    )
    annotation_status: Literal["none", "candidate", "annotated"] = Field(
        default="none",
        description="Annotation workflow state for this sample.",
    )
    annotated_label: int | None = Field(
        default=None,
        description="Human-provided label written by annotation job when available.",
    )
