# tests/unit/test_predict_record.py
"""
Unit tests for the PredictRecord Pydantic schema.

These tests cover field validation rules — particularly the embedding length
constraint (must be exactly EMBEDDING_DIM=64 elements) and the annotation
status enum, which together guard the DB insertion path.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from shared.schemas.feature_schema import EMBEDDING_DIM
from shared.schemas.predict_record import PredictRecord


def _valid_record(**overrides) -> dict:
    """Return kwargs that produce a valid PredictRecord."""
    base = {
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "model_version": "run-abc123",
        "embedding": [0.1] * EMBEDDING_DIM,
        "prediction": 3,
        "confidence": 0.87,
        "prediction_distribution": [0.1] * 10,
    }
    base.update(overrides)
    return base


class TestEmbeddingValidation:
    def test_exact_length_accepted(self):
        rec = PredictRecord(**_valid_record())
        assert len(rec.embedding) == EMBEDDING_DIM

    def test_too_short_rejected(self):
        with pytest.raises(ValidationError):
            PredictRecord(**_valid_record(embedding=[0.0] * (EMBEDDING_DIM - 1)))

    def test_too_long_rejected(self):
        with pytest.raises(ValidationError):
            PredictRecord(**_valid_record(embedding=[0.0] * (EMBEDDING_DIM + 1)))

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            PredictRecord(**_valid_record(embedding=[]))


class TestAnnotationStatus:
    def test_default_is_none(self):
        rec = PredictRecord(**_valid_record())
        assert rec.annotation_status == "none"

    def test_candidate_accepted(self):
        rec = PredictRecord(**_valid_record(annotation_status="candidate"))
        assert rec.annotation_status == "candidate"

    def test_annotated_accepted(self):
        rec = PredictRecord(**_valid_record(annotation_status="annotated"))
        assert rec.annotation_status == "annotated"

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            PredictRecord(**_valid_record(annotation_status="pending"))


class TestUuidField:
    def test_auto_generated_when_omitted(self):
        r1 = PredictRecord(**_valid_record())
        r2 = PredictRecord(**_valid_record())
        assert isinstance(r1.uuid, UUID)
        assert r1.uuid != r2.uuid

    def test_provided_uuid_preserved(self):
        uid = uuid4()
        rec = PredictRecord(**_valid_record(uuid=uid))
        assert rec.uuid == uid


class TestAnnotatedLabel:
    def test_defaults_to_none(self):
        rec = PredictRecord(**_valid_record())
        assert rec.annotated_label is None

    def test_integer_value_accepted(self):
        rec = PredictRecord(**_valid_record(annotated_label=7))
        assert rec.annotated_label == 7


class TestModelCopy:
    def test_copy_is_independent(self):
        """model_copy() used by FakeDataController must produce an independent object."""
        rec = PredictRecord(**_valid_record())
        copy = rec.model_copy()
        copy.prediction = 99
        assert rec.prediction == 3
