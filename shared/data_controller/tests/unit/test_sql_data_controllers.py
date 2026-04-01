from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pytest

from shared.data_controller._base import DataControllerError
from shared.data_controller.annotation import AnnotationDataController
from shared.data_controller.drift import DriftDataController
from shared.data_controller.sampling import SamplingDataController
from shared.data_controller.serving import ServingDataController
from shared.schemas.predict_record import PredictRecord


class _FakeCursor:
    def __init__(
        self, *, fetchall_rows=None, fetchone_row=None, execute_error: Exception | None = None
    ):
        self.fetchall_rows = fetchall_rows or []
        self.fetchone_row = fetchone_row
        self.execute_error = execute_error
        self.executed: list[tuple[str, tuple | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        if self.execute_error is not None:
            raise self.execute_error
        self.executed.append((sql, params))

    def fetchall(self):
        return self.fetchall_rows

    def fetchone(self):
        return self.fetchone_row


class _FakeConn:
    def __init__(self, cursor: _FakeCursor, *, rollback_error: Exception | None = None):
        self._cursor = cursor
        self.commit_calls = 0
        self.rollback_calls = 0
        self.rollback_error = rollback_error

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1
        if self.rollback_error is not None:
            raise self.rollback_error


def _record() -> PredictRecord:
    return PredictRecord(
        timestamp=datetime.now(UTC),
        model_version="v1",
        prediction=3,
        confidence=0.91,
        prediction_distribution=[0.01] * 10,
        embedding=[0.1] * 64,
        annotation_status="none",
        annotated_label=None,
    )


def test_annotation_get_candidates_returns_uuids_and_uses_limit() -> None:
    expected = [uuid4(), uuid4()]
    cur = _FakeCursor(fetchall_rows=[(expected[0],), (expected[1],)])
    conn = _FakeConn(cur)

    ctrl: Any = cast(Any, AnnotationDataController.__new__(AnnotationDataController))
    ctrl._connect = lambda: conn

    got = ctrl.get_candidates(limit=2)

    assert got == expected
    assert cur.executed[0][1] == (2,)


def test_annotation_write_label_rolls_back_and_wraps_error() -> None:
    target_uuid = uuid4()
    cur = _FakeCursor(execute_error=RuntimeError("db down"))
    conn = _FakeConn(cur)

    ctrl: Any = cast(Any, AnnotationDataController.__new__(AnnotationDataController))
    ctrl._conn = conn
    ctrl._connect = lambda: conn

    with pytest.raises(DataControllerError, match="Failed to write label"):
        ctrl.write_label(target_uuid, 7)

    assert conn.rollback_calls == 1


def test_sampling_select_and_mark_candidates_commits_and_returns_rows() -> None:
    expected = [uuid4(), uuid4(), uuid4()]
    cur = _FakeCursor(fetchall_rows=[(u,) for u in expected])
    conn = _FakeConn(cur)

    ctrl: Any = cast(Any, SamplingDataController.__new__(SamplingDataController))
    ctrl._connect = lambda: conn

    got = ctrl.select_and_mark_candidates(limit=3)

    assert got == expected
    assert conn.commit_calls == 1
    assert cur.executed[0][1] == (3,)


def test_sampling_select_and_mark_candidates_sets_conn_none_if_rollback_fails() -> None:
    cur = _FakeCursor(execute_error=RuntimeError("query failed"))
    conn = _FakeConn(cur, rollback_error=RuntimeError("rollback failed"))

    ctrl: Any = cast(Any, SamplingDataController.__new__(SamplingDataController))
    ctrl._conn = conn
    ctrl._connect = lambda: conn

    with pytest.raises(DataControllerError, match="Failed to mark candidates"):
        ctrl.select_and_mark_candidates(limit=5)

    assert ctrl._conn is None


def test_drift_get_annotated_count_wraps_db_errors() -> None:
    cur = _FakeCursor(execute_error=RuntimeError("count failed"))
    conn = _FakeConn(cur)

    ctrl: Any = cast(Any, DriftDataController.__new__(DriftDataController))
    ctrl._connect = lambda: conn

    with pytest.raises(DataControllerError, match="Failed to count annotated predictions"):
        ctrl.get_annotated_count()


def test_drift_get_predictions_returns_predict_records_and_passes_filters() -> None:
    now = datetime.now(UTC)
    row = (
        uuid4(),
        now,
        "v2",
        4,
        0.88,
        [0.0] * 10,
        [0.2] * 64,
        "none",
        None,
    )
    cur = _FakeCursor(fetchall_rows=[row])
    conn = _FakeConn(cur)

    ctrl: Any = cast(Any, DriftDataController.__new__(DriftDataController))
    ctrl._connect = lambda: conn

    since = datetime(2026, 1, 1, tzinfo=UTC)
    until = datetime(2026, 1, 2, tzinfo=UTC)
    records = ctrl.get_predictions(since=since, until=until, model_version="v2")

    assert len(records) == 1
    assert records[0].model_version == "v2"
    assert records[0].prediction == 4
    assert cur.executed[0][1] == (since, until, until, "v2", "v2")


def test_drift_get_labeled_predictions_returns_only_mapped_records() -> None:
    now = datetime.now(UTC)
    row = (
        uuid4(),
        now,
        "v3",
        2,
        0.77,
        [0.0] * 10,
        [0.3] * 64,
        "annotated",
        2,
    )
    cur = _FakeCursor(fetchall_rows=[row])
    conn = _FakeConn(cur)

    ctrl: Any = cast(Any, DriftDataController.__new__(DriftDataController))
    ctrl._connect = lambda: conn

    since = datetime(2026, 1, 1, tzinfo=UTC)
    records = ctrl.get_labeled_predictions(since=since)

    assert len(records) == 1
    assert records[0].annotation_status == "annotated"
    assert records[0].annotated_label == 2
    assert cur.executed[0][1] == (since,)


def test_serving_store_prediction_swallow_db_error_and_still_upload_image() -> None:
    rec = _record()
    image = [[0.25 for _ in range(14)] for _ in range(14)]

    cur = _FakeCursor(execute_error=RuntimeError("insert failed"))
    conn = _FakeConn(cur)

    class _Store:
        def __init__(self):
            self.calls = []

        def put_array(self, key, arr):
            self.calls.append((key, arr))

    store = _Store()

    ctrl: Any = cast(Any, ServingDataController.__new__(ServingDataController))
    ctrl._available = True
    ctrl._s3_available = True
    ctrl._failures = 0
    ctrl._conn = conn
    ctrl._connect = lambda: conn
    ctrl._store = store

    # Fire-and-forget semantics: should not raise even if DB insert fails.
    ctrl.store_prediction(rec, image_2d=image)

    assert ctrl._failures == 1
    assert conn.rollback_calls == 1

    assert len(store.calls) == 1
    key, arr = store.calls[0]
    assert key == f"predictions/{rec.uuid}.npy"
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (14, 14)
