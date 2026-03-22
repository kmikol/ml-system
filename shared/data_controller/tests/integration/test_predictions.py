# shared/data_controller/tests/integration/test_predictions.py
"""
Integration tests for the prediction-table controllers (Serving, Drift, Sampling, Annotation).

All four controllers operate on the same `predictions` table. Tests use unique
UUIDs so they are independent of each other and of whatever else is in the table.

Requires Docker (Postgres container started by conftest.py).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

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
    label: int | None = None,
    annotation_status: str = "none",
    ts: datetime | None = None,
) -> PredictRecord:
    return PredictRecord(
        prediction_id=str(uuid.uuid4()),
        timestamp=ts or datetime.now(UTC),
        model_version=model_version,
        image=[[0.0] * 14] * 14,
        embedding=[0.0] * 32,
        prediction=3,
        confidence=0.87,
        prediction_distribution=[0.1] * 10,
        label=label,
        annotation_status=annotation_status,
    )


# ── ServingDataController ─────────────────────────────────────────────────────


class TestServingDataController:
    def test_store_prediction_is_readable(self):
        record = _record()
        ServingDataController().store_prediction(record)

        results = DriftDataController().get_predictions(since=_EPOCH)
        assert any(r.prediction_id == record.prediction_id for r in results)

    def test_store_prediction_persists_all_fields(self):
        record = _record(model_version="v42")
        ServingDataController().store_prediction(record)

        results = DriftDataController().get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.prediction_id == record.prediction_id)

        assert stored.model_version == "v42"
        assert stored.prediction == record.prediction
        assert stored.confidence == pytest.approx(record.confidence)
        assert stored.label is None
        assert stored.annotation_status == "none"

    def test_store_prediction_is_idempotent(self):
        record = _record()
        ctrl = ServingDataController()
        ctrl.store_prediction(record)
        ctrl.store_prediction(record)  # ON CONFLICT DO NOTHING

        results = DriftDataController().get_predictions(since=_EPOCH)
        count = sum(1 for r in results if r.prediction_id == record.prediction_id)
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
        ids = {r.prediction_id for r in results}
        assert new.prediction_id in ids
        assert old.prediction_id not in ids

    def test_get_predictions_until_bound(self):
        now = datetime.now(UTC)
        before = _record(ts=now - timedelta(minutes=30))
        after = _record(ts=now + timedelta(minutes=30))
        serving = ServingDataController()
        serving.store_prediction(before)
        serving.store_prediction(after)

        drift = DriftDataController()
        results = drift.get_predictions(since=_EPOCH, until=now)
        ids = {r.prediction_id for r in results}
        assert before.prediction_id in ids
        assert after.prediction_id not in ids

    def test_get_predictions_model_version_filter(self):
        r1 = _record(model_version="filter-v1")
        r2 = _record(model_version="filter-v2")
        serving = ServingDataController()
        serving.store_prediction(r1)
        serving.store_prediction(r2)

        drift = DriftDataController()
        results = drift.get_predictions(since=_EPOCH, model_version="filter-v1")
        ids = {r.prediction_id for r in results}
        assert r1.prediction_id in ids
        assert r2.prediction_id not in ids

    def test_get_labeled_predictions(self):
        labeled = _record(label=5)
        unlabeled = _record()
        serving = ServingDataController()
        serving.store_prediction(labeled)
        serving.store_prediction(unlabeled)

        # Write a real label via AnnotationDataController so the DB row is updated
        AnnotationDataController().write_label(labeled.prediction_id, 5)

        drift = DriftDataController()
        results = drift.get_labeled_predictions(since=_EPOCH)
        ids = {r.prediction_id for r in results}
        assert labeled.prediction_id in ids
        assert unlabeled.prediction_id not in ids


# ── SamplingDataController ────────────────────────────────────────────────────


class TestSamplingDataController:
    def test_mark_candidate_advances_status(self):
        record = _record()
        ServingDataController().store_prediction(record)

        sampling = SamplingDataController()
        sampling.mark_candidate(record.prediction_id)

        drift = DriftDataController()
        results = drift.get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.prediction_id == record.prediction_id)
        assert stored.annotation_status == "candidate"

    def test_mark_candidate_is_idempotent_when_already_annotated(self):
        record = _record()
        ServingDataController().store_prediction(record)

        annotation = AnnotationDataController()
        annotation.write_label(record.prediction_id, 7)

        sampling = SamplingDataController()
        sampling.mark_candidate(record.prediction_id)  # must not raise

        drift = DriftDataController()
        results = drift.get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.prediction_id == record.prediction_id)
        assert stored.annotation_status == "annotated"  # unchanged

    def test_count_labels_since(self):
        now = datetime.now(UTC)
        r1 = _record(ts=now)
        r2 = _record(ts=now)
        serving = ServingDataController()
        serving.store_prediction(r1)
        serving.store_prediction(r2)

        annotation = AnnotationDataController()
        annotation.write_label(r1.prediction_id, 1)
        annotation.write_label(r2.prediction_id, 2)

        sampling = SamplingDataController()
        count = sampling.count_labels_since(since=now - timedelta(seconds=1))
        assert count >= 2


# ── AnnotationDataController ──────────────────────────────────────────────────


class TestAnnotationDataController:
    def test_write_label_sets_label_and_annotated_status(self):
        record = _record()
        ServingDataController().store_prediction(record)

        AnnotationDataController().write_label(record.prediction_id, 9)

        drift = DriftDataController()
        results = drift.get_predictions(since=_EPOCH)
        stored = next(r for r in results if r.prediction_id == record.prediction_id)
        assert stored.label == 9
        assert stored.annotation_status == "annotated"


# ── End-to-end workflow ───────────────────────────────────────────────────────


class TestFullWorkflow:
    def test_store_sample_mark_annotate_query(self):
        """Serve → drift query → mark candidate → annotate → labeled query."""
        record = _record()

        # 1. Serving stores prediction
        ServingDataController().store_prediction(record)

        # 2. Drift can see it
        drift = DriftDataController()
        assert any(r.prediction_id == record.prediction_id for r in drift.get_predictions(since=_EPOCH))

        # 3. Sampling marks it as a candidate
        SamplingDataController().mark_candidate(record.prediction_id)
        after_mark = next(r for r in drift.get_predictions(since=_EPOCH) if r.prediction_id == record.prediction_id)
        assert after_mark.annotation_status == "candidate"

        # 4. Annotation writes a label
        AnnotationDataController().write_label(record.prediction_id, 4)

        # 5. Drift sees it in labeled query
        labeled = drift.get_labeled_predictions(since=_EPOCH)
        annotated = next((r for r in labeled if r.prediction_id == record.prediction_id), None)
        assert annotated is not None
        assert annotated.label == 4
        assert annotated.annotation_status == "annotated"
