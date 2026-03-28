# tests/unit/test_schemas.py
"""
Unit tests for shared/schemas/api.py and shared/schemas/inference_event.py.

No env vars or external services needed — pure Pydantic validation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from shared.schemas.api import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ValidationErrorResponse,
)
from shared.schemas.inference_event import InferenceEvent

_VALID_IMAGE = [[0.5] * 14 for _ in range(14)]
_UUID_STR = "12345678-1234-5678-1234-567812345678"


# ── PredictRequest ────────────────────────────────────────────────────────────


class TestPredictRequest:
    def test_valid_image_accepted(self):
        req = PredictRequest(image=_VALID_IMAGE)
        assert req.image == _VALID_IMAGE

    def test_uuid_defaults_to_none(self):
        req = PredictRequest(image=_VALID_IMAGE)
        assert req.uuid is None

    def test_uuid_accepted_as_string(self):
        req = PredictRequest(image=_VALID_IMAGE, uuid=_UUID_STR)
        assert isinstance(req.uuid, UUID)
        assert str(req.uuid) == _UUID_STR

    def test_uuid_accepted_as_uuid_object(self):
        uid = uuid4()
        req = PredictRequest(image=_VALID_IMAGE, uuid=uid)
        assert req.uuid == uid

    def test_missing_image_raises(self):
        with pytest.raises(ValidationError):
            PredictRequest()

    def test_image_preserves_values(self):
        image = [[float(i * 14 + j) / 196 for j in range(14)] for i in range(14)]
        req = PredictRequest(image=image)
        assert req.image[0][0] == pytest.approx(0.0)
        assert req.image[13][13] == pytest.approx(195.0 / 196)


# ── PredictResponse ───────────────────────────────────────────────────────────


class TestPredictResponse:
    def test_serializes_all_fields(self):
        uid = uuid4()
        resp = PredictResponse(prediction=3, confidence=0.85, model_version="run-1", uuid=uid)
        d = resp.model_dump()
        assert d["prediction"] == 3
        assert d["confidence"] == pytest.approx(0.85)
        assert d["model_version"] == "run-1"
        assert d["uuid"] == uid

    def test_prediction_is_int(self):
        resp = PredictResponse(prediction=0, confidence=1.0, model_version="v1", uuid=uuid4())
        assert isinstance(resp.prediction, int)

    def test_confidence_is_float(self):
        resp = PredictResponse(prediction=5, confidence=0.5, model_version="v1", uuid=uuid4())
        assert isinstance(resp.confidence, float)

    def test_uuid_field_is_uuid(self):
        uid = uuid4()
        resp = PredictResponse(prediction=1, confidence=0.9, model_version="v1", uuid=uid)
        assert isinstance(resp.uuid, UUID)


# ── HealthResponse ────────────────────────────────────────────────────────────


class TestHealthResponse:
    def test_model_version_is_optional(self):
        resp = HealthResponse(status="unhealthy", model_loaded=False, uptime_seconds=1.0)
        assert resp.model_version is None

    def test_model_version_set_when_provided(self):
        resp = HealthResponse(
            status="healthy", model_loaded=True, model_version="run-42", uptime_seconds=10.5
        )
        assert resp.model_version == "run-42"

    def test_uptime_seconds_is_float(self):
        resp = HealthResponse(status="healthy", model_loaded=True, uptime_seconds=3.7)
        assert isinstance(resp.uptime_seconds, float)

    def test_model_loaded_is_bool(self):
        resp = HealthResponse(status="healthy", model_loaded=True, uptime_seconds=0.0)
        assert resp.model_loaded is True

    def test_serializes_to_dict(self):
        resp = HealthResponse(status="healthy", model_loaded=True, uptime_seconds=5.0)
        d = resp.model_dump()
        assert "status" in d
        assert "model_loaded" in d
        assert "uptime_seconds" in d
        assert "model_version" in d


# ── ValidationErrorResponse ───────────────────────────────────────────────────


class TestValidationErrorResponse:
    def test_detail_field(self):
        resp = ValidationErrorResponse(detail="Validation failed", errors=[])
        assert resp.detail == "Validation failed"

    def test_errors_field_is_list(self):
        errors = [{"field": "image", "msg": "wrong size"}]
        resp = ValidationErrorResponse(detail="bad", errors=errors)
        assert resp.errors == errors

    def test_empty_errors_allowed(self):
        resp = ValidationErrorResponse(detail="ok", errors=[])
        assert resp.errors == []


# ── InferenceEvent ─────────────────────────────────────────────────────────────


class TestInferenceEvent:
    def _make_event(self, **overrides) -> InferenceEvent:
        defaults = {
            "event_id": "evt-1",
            "timestamp": datetime.now(UTC),
            "model_version": "run-1",
            "request_id": "req-1",
            "image": _VALID_IMAGE,
            "embedding": [0.0] * 64,
            "prediction": 3,
            "confidence": 0.9,
            "prediction_distribution": [0.1] * 10,
        }
        defaults.update(overrides)
        return InferenceEvent(**defaults)

    def test_valid_construction(self):
        event = self._make_event()
        assert event.event_id == "evt-1"
        assert event.prediction == 3

    def test_schema_version_defaults_to_1_0(self):
        event = self._make_event()
        assert event.schema_version == "1.0"

    def test_schema_version_overridable(self):
        event = self._make_event(schema_version="2.0")
        assert event.schema_version == "2.0"

    def test_all_fields_present_in_dump(self):
        event = self._make_event()
        d = event.model_dump()
        for field in [
            "event_id",
            "timestamp",
            "model_version",
            "request_id",
            "image",
            "embedding",
            "prediction",
            "confidence",
            "prediction_distribution",
            "schema_version",
        ]:
            assert field in d, f"Missing field: {field}"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            InferenceEvent(
                timestamp=datetime.now(UTC),
                model_version="v1",
                # missing event_id and other required fields
            )
