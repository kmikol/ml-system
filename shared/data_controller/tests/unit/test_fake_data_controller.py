"""Unit tests for FakeDataController.

No Postgres or external services required — FakeDataController is fully in-memory.
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from shared.data_controller.fake import FakeDataController
from shared.schemas.predict_record import PredictRecord


def _record(
    uuid: UUID | None = None,
    timestamp: datetime | None = None,
    model_version: str = "v1",
    annotated_label: int | None = None,
    annotation_status: Literal["none", "candidate", "annotated"] = "none",
) -> PredictRecord:
    return PredictRecord(
        uuid=uuid or uuid4(),
        timestamp=timestamp or datetime(2026, 1, 1, tzinfo=UTC),
        model_version=model_version,
        embedding=[0.0] * 64,
        prediction=0,
        confidence=0.9,
        prediction_distribution=[0.1] * 10,
        annotated_label=annotated_label,
        annotation_status=annotation_status,
    )


# ── store_prediction ──────────────────────────────────────────────────────────


class TestStorePrediction:
    def test_stores_record(self):
        ctrl = FakeDataController()
        r = _record()
        ctrl.store_prediction(r)
        assert len(ctrl._records) == 1

    def test_stores_a_copy(self):
        ctrl = FakeDataController()
        r = _record()
        ctrl.store_prediction(r)
        r.prediction = 99
        assert ctrl._records[0].prediction == 0


# ── get_predictions ───────────────────────────────────────────────────────────


class TestGetPredictions:
    def test_returns_records_since(self):
        ctrl = FakeDataController()
        u1 = uuid4()
        u2 = uuid4()
        ctrl.store_prediction(_record(u1, datetime(2026, 1, 1, tzinfo=UTC)))
        ctrl.store_prediction(_record(u2, datetime(2026, 1, 2, tzinfo=UTC)))

        results = ctrl.get_predictions(since=datetime(2026, 1, 2, tzinfo=UTC))
        assert [r.uuid for r in results] == [u2]

    def test_filters_by_until(self):
        ctrl = FakeDataController()
        u1 = uuid4()
        u2 = uuid4()
        ctrl.store_prediction(_record(u1, datetime(2026, 1, 1, tzinfo=UTC)))
        ctrl.store_prediction(_record(u2, datetime(2026, 1, 3, tzinfo=UTC)))

        results = ctrl.get_predictions(
            since=datetime(2026, 1, 1, tzinfo=UTC),
            until=datetime(2026, 1, 2, tzinfo=UTC),
        )
        assert [r.uuid for r in results] == [u1]

    def test_filters_by_model_version(self):
        ctrl = FakeDataController()
        u1 = uuid4()
        u2 = uuid4()
        ctrl.store_prediction(_record(u1, model_version="v1"))
        ctrl.store_prediction(_record(u2, model_version="v2"))

        results = ctrl.get_predictions(
            since=datetime(2026, 1, 1, tzinfo=UTC),
            model_version="v2",
        )
        assert [r.uuid for r in results] == [u2]

    def test_returns_empty_when_no_match(self):
        ctrl = FakeDataController()
        results = ctrl.get_predictions(since=datetime(2026, 1, 1, tzinfo=UTC))
        assert results == []


# ── get_labeled_predictions ───────────────────────────────────────────────────


class TestGetLabeledPredictions:
    def test_returns_only_annotated_labeled(self):
        ctrl = FakeDataController()
        u1 = uuid4()
        u2 = uuid4()
        ctrl.store_prediction(_record(u1, annotated_label=None))
        ctrl.store_prediction(_record(u2, annotated_label=3))

        results = ctrl.get_labeled_predictions(since=datetime(2026, 1, 1, tzinfo=UTC))
        assert [r.uuid for r in results] == [u2]

    def test_filters_by_since(self):
        ctrl = FakeDataController()
        u1 = uuid4()
        u2 = uuid4()
        ctrl.store_prediction(_record(u1, datetime(2026, 1, 1, tzinfo=UTC), annotated_label=1))
        ctrl.store_prediction(_record(u2, datetime(2026, 1, 3, tzinfo=UTC), annotated_label=2))

        results = ctrl.get_labeled_predictions(since=datetime(2026, 1, 2, tzinfo=UTC))
        assert [r.uuid for r in results] == [u2]


# ── mark_candidate ────────────────────────────────────────────────────────────


class TestMarkCandidate:
    def test_advances_status_to_candidate(self):
        ctrl = FakeDataController()
        uid = uuid4()
        ctrl.store_prediction(_record(uid))
        ctrl.mark_candidate(uid)
        assert ctrl._records[0].annotation_status == "candidate"

    def test_ignores_already_annotated(self):
        ctrl = FakeDataController()
        uid = uuid4()
        ctrl.store_prediction(_record(uid, annotation_status="annotated"))
        ctrl.mark_candidate(uid)
        assert ctrl._records[0].annotation_status == "annotated"

    def test_ignores_unknown_id(self):
        ctrl = FakeDataController()
        ctrl.mark_candidate(uuid4())  # must not raise


# ── select_and_mark_candidates ────────────────────────────────────────────────


class TestSelectAndMarkCandidates:
    def test_marks_eligible_predictions(self):
        ctrl = FakeDataController()
        uid = uuid4()
        ctrl.store_prediction(_record(uid))

        marked = ctrl.select_and_mark_candidates(limit=10)

        assert uid in marked
        assert ctrl._records[0].annotation_status == "candidate"

    def test_skips_already_candidate_or_annotated(self):
        ctrl = FakeDataController()
        u1 = uuid4()
        u2 = uuid4()
        ctrl.store_prediction(_record(u1, annotation_status="candidate"))
        ctrl.store_prediction(_record(u2, annotation_status="annotated"))

        marked = ctrl.select_and_mark_candidates(limit=10)
        assert marked == []

    def test_respects_limit(self):
        ctrl = FakeDataController()
        for _ in range(5):
            ctrl.store_prediction(_record())

        marked = ctrl.select_and_mark_candidates(limit=3)
        assert len(marked) == 3
        candidate_count = sum(1 for r in ctrl._records if r.annotation_status == "candidate")
        assert candidate_count == 3


# ── write_label ───────────────────────────────────────────────────────────────


class TestWriteLabel:
    def test_sets_annotated_label_and_status(self):
        ctrl = FakeDataController()
        uid = uuid4()
        ctrl.store_prediction(_record(uid, annotation_status="candidate"))
        ctrl.write_label(uid, 7)
        assert ctrl._records[0].annotated_label == 7
        assert ctrl._records[0].annotation_status == "annotated"

    def test_no_op_when_not_candidate(self):
        ctrl = FakeDataController()
        uid = uuid4()
        ctrl.store_prediction(_record(uid))  # annotation_status='none'
        ctrl.write_label(uid, 7)
        assert ctrl._records[0].annotated_label is None
        assert ctrl._records[0].annotation_status == "none"

    def test_ignores_unknown_id(self):
        ctrl = FakeDataController()
        ctrl.write_label(uuid4(), 5)  # must not raise


# ── reset_candidate ───────────────────────────────────────────────────────────


class TestResetCandidate:
    def test_resets_candidate_to_none(self):
        ctrl = FakeDataController()
        uid = uuid4()
        ctrl.store_prediction(_record(uid, annotation_status="candidate"))
        ctrl.reset_candidate(uid)
        assert ctrl._records[0].annotation_status == "none"

    def test_no_op_when_not_candidate(self):
        ctrl = FakeDataController()
        uid = uuid4()
        ctrl.store_prediction(_record(uid, annotation_status="annotated"))
        ctrl.reset_candidate(uid)
        assert ctrl._records[0].annotation_status == "annotated"

    def test_ignores_unknown_id(self):
        ctrl = FakeDataController()
        ctrl.reset_candidate(uuid4())  # must not raise


# ── get_candidates ────────────────────────────────────────────────────────────


class TestGetCandidates:
    def test_returns_candidate_uuids(self):
        ctrl = FakeDataController()
        uid = uuid4()
        ctrl.store_prediction(_record(uid, annotation_status="candidate"))

        results = ctrl.get_candidates(limit=10)
        assert uid in results

    def test_excludes_non_candidate_status(self):
        ctrl = FakeDataController()
        u1 = uuid4()
        u2 = uuid4()
        ctrl.store_prediction(_record(u1, annotation_status="none"))
        ctrl.store_prediction(_record(u2, annotation_status="annotated"))

        results = ctrl.get_candidates(limit=10)
        assert results == []

    def test_respects_limit(self):
        ctrl = FakeDataController()
        for _ in range(5):
            ctrl.store_prediction(_record(annotation_status="candidate"))

        results = ctrl.get_candidates(limit=3)
        assert len(results) == 3

    def test_returns_empty_when_no_candidates(self):
        ctrl = FakeDataController()
        results = ctrl.get_candidates(limit=10)
        assert results == []
