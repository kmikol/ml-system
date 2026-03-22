"""Unit tests for FakeDataController.

No Postgres or external services required — FakeDataController is fully in-memory.
"""

from datetime import UTC, datetime

from shared.data_controller.fake import FakeDataController
from shared.schemas.predict_record import PredictRecord


def _record(
    prediction_id: str = "pred-1",
    timestamp: datetime | None = None,
    model_version: str = "v1",
    label: int | None = None,
    annotation_status: str = "none",
) -> PredictRecord:
    return PredictRecord(
        prediction_id=prediction_id,
        timestamp=timestamp or datetime(2026, 1, 1, tzinfo=UTC),
        model_version=model_version,
        image=[[0.0] * 14] * 14,
        embedding=[0.0] * 32,
        prediction=0,
        confidence=0.9,
        prediction_distribution=[0.1] * 10,
        label=label,
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
        ctrl.store_prediction(_record("p1", datetime(2026, 1, 1, tzinfo=UTC)))
        ctrl.store_prediction(_record("p2", datetime(2026, 1, 2, tzinfo=UTC)))

        results = ctrl.get_predictions(since=datetime(2026, 1, 2, tzinfo=UTC))
        assert [r.prediction_id for r in results] == ["p2"]

    def test_filters_by_until(self):
        ctrl = FakeDataController()
        ctrl.store_prediction(_record("p1", datetime(2026, 1, 1, tzinfo=UTC)))
        ctrl.store_prediction(_record("p2", datetime(2026, 1, 3, tzinfo=UTC)))

        results = ctrl.get_predictions(
            since=datetime(2026, 1, 1, tzinfo=UTC),
            until=datetime(2026, 1, 2, tzinfo=UTC),
        )
        assert [r.prediction_id for r in results] == ["p1"]

    def test_filters_by_model_version(self):
        ctrl = FakeDataController()
        ctrl.store_prediction(_record("p1", model_version="v1"))
        ctrl.store_prediction(_record("p2", model_version="v2"))

        results = ctrl.get_predictions(
            since=datetime(2026, 1, 1, tzinfo=UTC),
            model_version="v2",
        )
        assert [r.prediction_id for r in results] == ["p2"]

    def test_returns_empty_when_no_match(self):
        ctrl = FakeDataController()
        results = ctrl.get_predictions(since=datetime(2026, 1, 1, tzinfo=UTC))
        assert results == []


# ── get_labeled_predictions ───────────────────────────────────────────────────


class TestGetLabeledPredictions:
    def test_returns_only_labeled(self):
        ctrl = FakeDataController()
        ctrl.store_prediction(_record("p1", label=None))
        ctrl.store_prediction(_record("p2", label=3))

        results = ctrl.get_labeled_predictions(since=datetime(2026, 1, 1, tzinfo=UTC))
        assert [r.prediction_id for r in results] == ["p2"]

    def test_filters_by_since(self):
        ctrl = FakeDataController()
        ctrl.store_prediction(_record("p1", datetime(2026, 1, 1, tzinfo=UTC), label=1))
        ctrl.store_prediction(_record("p2", datetime(2026, 1, 3, tzinfo=UTC), label=2))

        results = ctrl.get_labeled_predictions(since=datetime(2026, 1, 2, tzinfo=UTC))
        assert [r.prediction_id for r in results] == ["p2"]


# ── mark_candidate ────────────────────────────────────────────────────────────


class TestMarkCandidate:
    def test_advances_status_to_candidate(self):
        ctrl = FakeDataController()
        ctrl.store_prediction(_record("p1"))
        ctrl.mark_candidate("p1")
        assert ctrl._records[0].annotation_status == "candidate"

    def test_ignores_already_annotated(self):
        ctrl = FakeDataController()
        ctrl.store_prediction(_record("p1", annotation_status="annotated"))
        ctrl.mark_candidate("p1")
        assert ctrl._records[0].annotation_status == "annotated"

    def test_ignores_unknown_id(self):
        ctrl = FakeDataController()
        ctrl.mark_candidate("nonexistent")  # must not raise


# ── write_label ───────────────────────────────────────────────────────────────


class TestWriteLabel:
    def test_sets_label_and_annotated_status(self):
        ctrl = FakeDataController()
        ctrl.store_prediction(_record("p1"))
        ctrl.write_label("p1", 7)
        assert ctrl._records[0].label == 7
        assert ctrl._records[0].annotation_status == "annotated"

    def test_ignores_unknown_id(self):
        ctrl = FakeDataController()
        ctrl.write_label("nonexistent", 5)  # must not raise


# ── count_labels_since ────────────────────────────────────────────────────────


class TestCountLabelsSince:
    def test_counts_labeled_records_since(self):
        ctrl = FakeDataController()
        ctrl.store_prediction(_record("p1", datetime(2026, 1, 1, tzinfo=UTC), label=1))
        ctrl.store_prediction(_record("p2", datetime(2026, 1, 3, tzinfo=UTC), label=2))
        ctrl.store_prediction(_record("p3", datetime(2026, 1, 3, tzinfo=UTC), label=None))

        count = ctrl.count_labels_since(since=datetime(2026, 1, 2, tzinfo=UTC))
        assert count == 1

    def test_returns_zero_when_none(self):
        ctrl = FakeDataController()
        count = ctrl.count_labels_since(since=datetime(2026, 1, 1, tzinfo=UTC))
        assert count == 0
