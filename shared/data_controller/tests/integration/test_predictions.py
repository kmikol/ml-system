# shared/data_controller/tests/integration/test_predictions.py
"""
Integration tests for the prediction-table controllers (Serving, Drift, Sampling, Annotation).

All four controllers operate on the same ``predictions`` table.  Tests use unique
UUIDs so they are independent of each other and of whatever else is in the table.

Requires Docker (Postgres + MinIO containers defined in docker-compose.test.yml).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from shared.data_controller.annotation import AnnotationDataController
from shared.data_controller.drift import DriftDataController
from shared.data_controller.sampling import SamplingDataController
from shared.data_controller.serving import ServingDataController
from shared.schemas.predict_record import PredictRecord

# A timestamp safely in the past so "since" queries catch all test records.
_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


def _record(
    model_version: str = "v1",
    annotation_status: str = "none",
    ts: datetime | None = None,
) -> PredictRecord:
    """Build a PredictRecord with a fresh UUID."""
    return PredictRecord(
        timestamp=ts or datetime.now(UTC),
        model_version=model_version,
        embedding=[0.0] * 32,
        prediction=3,
        confidence=0.87,
        prediction_distribution=[0.1] * 10,
        annotation_status=annotation_status,
    )


# ── ServingDataController ─────────────────────────────────────────────────────


class TestServingDataController:
    def test_store_prediction_is_readable(self):
        record = _record()
        ServingDataController().store_prediction(record)

        results = DriftDataController().get_predictions(since=_EPOCH)
        assert any(r.uuid == record.uuid for r in results)

    def test_store_prediction_persists_all_fields(self):
        record = _record(model_version="v42")
        ServingDataController().store_prediction(record)

        results = DriftDataController().get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.uuid == record.uuid)

        assert stored.model_version == "v42"
        assert stored.prediction == record.prediction
        assert stored.confidence == pytest.approx(record.confidence)
        assert stored.annotated_label is None
        assert stored.annotation_status == "none"

    def test_store_prediction_is_idempotent(self):
        record = _record()
        ctrl = ServingDataController()
        ctrl.store_prediction(record)
        ctrl.store_prediction(record)  # ON CONFLICT DO NOTHING

        results = DriftDataController().get_predictions(since=_EPOCH)
        count = sum(1 for r in results if r.uuid == record.uuid)
        assert count == 1


# ── DriftDataController ───────────────────────────────────────────────────────


class TestDriftDataController:
    def test_get_predictions_time_window(self):
        now = datetime.now(UTC)
        old = _record(ts=now - timedelta(hours=2))
        new = _record(ts=now)
        ctrl = ServingDataController()
        ctrl.store_prediction(old)
        ctrl.store_prediction(new)

        drift = DriftDataController()
        results = drift.get_predictions(since=now - timedelta(hours=1))
        ids = {r.uuid for r in results}
        assert new.uuid in ids
        assert old.uuid not in ids

    def test_get_predictions_until_bound(self):
        now = datetime.now(UTC)
        before = _record(ts=now - timedelta(minutes=30))
        after = _record(ts=now + timedelta(minutes=30))
        serving = ServingDataController()
        serving.store_prediction(before)
        serving.store_prediction(after)

        drift = DriftDataController()
        results = drift.get_predictions(since=_EPOCH, until=now)
        ids = {r.uuid for r in results}
        assert before.uuid in ids
        assert after.uuid not in ids

    def test_get_predictions_model_version_filter(self):
        r1 = _record(model_version="filter-v1")
        r2 = _record(model_version="filter-v2")
        serving = ServingDataController()
        serving.store_prediction(r1)
        serving.store_prediction(r2)

        drift = DriftDataController()
        results = drift.get_predictions(since=_EPOCH, model_version="filter-v1")
        ids = {r.uuid for r in results}
        assert r1.uuid in ids
        assert r2.uuid not in ids

    def test_get_labeled_predictions(self):
        record = _record()
        unlabeled = _record()
        serving = ServingDataController()
        serving.store_prediction(record)
        serving.store_prediction(unlabeled)

        # Write a label via AnnotationDataController so the DB row is updated
        AnnotationDataController().write_label(record.uuid, 5)

        drift = DriftDataController()
        results = drift.get_labeled_predictions(since=_EPOCH)
        ids = {r.uuid for r in results}
        assert record.uuid in ids
        assert unlabeled.uuid not in ids


# ── SamplingDataController ────────────────────────────────────────────────────


class TestSamplingDataController:
    def test_select_and_mark_candidates_marks_eligible_predictions(self):
        """Any prediction with annotation_status='none' is eligible."""
        record = _record()
        ServingDataController().store_prediction(record)

        sampling = SamplingDataController()
        marked = sampling.select_and_mark_candidates(limit=100)

        assert record.uuid in marked

        results = DriftDataController().get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.uuid == record.uuid)
        assert stored.annotation_status == "candidate"

    def test_select_and_mark_candidates_is_idempotent_when_already_annotated(self):
        """Already-annotated predictions are not downgraded by a second sampling run."""
        record = _record()
        ServingDataController().store_prediction(record)
        AnnotationDataController().write_label(record.uuid, 7)

        # Sampling must not overwrite 'annotated' status
        SamplingDataController().select_and_mark_candidates(limit=100)

        results = DriftDataController().get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.uuid == record.uuid)
        assert stored.annotation_status == "annotated"  # unchanged

    def test_select_and_mark_candidates_skips_already_candidate(self):
        """Predictions already marked as candidate are not re-marked."""
        record = _record(annotation_status="candidate")
        ServingDataController().store_prediction(record)

        SamplingDataController().select_and_mark_candidates(limit=100)

        results = DriftDataController().get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.uuid == record.uuid)
        assert stored.annotation_status == "candidate"  # unchanged


# ── AnnotationDataController ──────────────────────────────────────────────────


class TestAnnotationDataController:
    def test_write_label_sets_annotated_label_and_status(self):
        record = _record()
        ServingDataController().store_prediction(record)

        AnnotationDataController().write_label(record.uuid, 9)

        drift = DriftDataController()
        results = drift.get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.uuid == record.uuid)
        assert stored.annotated_label == 9
        assert stored.annotation_status == "annotated"

    def test_get_candidates_returns_candidate_uuids(self):
        """get_candidates() returns UUIDs of predictions with annotation_status='candidate'."""
        record = _record(annotation_status="candidate")
        ServingDataController().store_prediction(record)

        candidates = AnnotationDataController().get_candidates(limit=100)

        assert record.uuid in candidates

    def test_get_candidates_excludes_non_candidates(self):
        """Predictions with status 'none' or 'annotated' are not returned."""
        none_record = _record(annotation_status="none")
        annotated_record = _record(annotation_status="annotated")
        ServingDataController().store_prediction(none_record)
        ServingDataController().store_prediction(annotated_record)

        candidates = AnnotationDataController().get_candidates(limit=100)

        assert none_record.uuid not in candidates
        assert annotated_record.uuid not in candidates


# ── End-to-end workflow ───────────────────────────────────────────────────────


class TestFullWorkflow:
    def test_serve_sample_annotate_query(self):
        """Serve → drift query → select_and_mark_candidates → annotate → labeled query."""
        # 1. Serving stores a prediction with a known UUID
        record = _record()
        ServingDataController().store_prediction(record)

        # 2. Drift can see it
        drift = DriftDataController()
        assert any(r.uuid == record.uuid for r in drift.get_predictions(since=_EPOCH))

        # 3. Sampling marks it as a candidate
        marked = SamplingDataController().select_and_mark_candidates(limit=100)
        assert record.uuid in marked

        after_mark = next(
            r for r in drift.get_predictions(since=_EPOCH)
            if r.uuid == record.uuid
        )
        assert after_mark.annotation_status == "candidate"

        # 4. Annotation fetches the candidate UUID and writes a label
        candidates = AnnotationDataController().get_candidates(limit=100)
        assert record.uuid in candidates

        AnnotationDataController().write_label(record.uuid, 3)

        # 5. Drift sees it in the labeled query
        labeled = drift.get_labeled_predictions(since=_EPOCH)
        annotated = next((r for r in labeled if r.uuid == record.uuid), None)
        assert annotated is not None
        assert annotated.annotated_label == 3
        assert annotated.annotation_status == "annotated"
