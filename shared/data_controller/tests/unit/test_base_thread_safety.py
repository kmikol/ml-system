# shared/data_controller/tests/unit/test_base_thread_safety.py
"""Unit tests verifying thread-safety of _DataControllerBase.

No real Postgres connection is required — psycopg2 is fully mocked.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from shared.data_controller._base import _DataControllerBase


def _make_base(mock_conn: MagicMock | None = None) -> _DataControllerBase:
    """Return a _DataControllerBase instance with all Postgres calls mocked."""
    if mock_conn is None:
        mock_conn = MagicMock()
        mock_conn.closed = False

    mock_psycopg2 = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn

    with patch("psycopg2.connect", mock_psycopg2.connect), patch(
        "psycopg2.extras.register_uuid"
    ):
        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": MagicMock()}):
            base = object.__new__(_DataControllerBase)
            base._psycopg2 = mock_psycopg2
            base._conn = None
            base._conn_lock = threading.Lock()
            base._dsn = "postgresql://test"
    return base


class TestConnLockExists:
    def test_conn_lock_is_threading_lock(self):
        """_conn_lock must be a threading.Lock so it can be used as a context manager."""
        base = _make_base()
        assert isinstance(base._conn_lock, type(threading.Lock()))

    def test_conn_lock_initialised_before_ensure_schema(self):
        """_conn_lock must exist by the time _ensure_schema (and thus _connect) is called."""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_psycopg2.extras = MagicMock()

        lock_existed_during_connect = []

        original_connect = _DataControllerBase._connect

        def patched_connect(self):
            lock_existed_during_connect.append(hasattr(self, "_conn_lock"))
            return original_connect(self)

        with patch.dict(
            "sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras}
        ), patch.object(_DataControllerBase, "_connect", patched_connect):
            base = object.__new__(_DataControllerBase)
            _DataControllerBase.__init__(base, "postgresql://test")

        assert lock_existed_during_connect, "_connect was never called"
        assert all(lock_existed_during_connect), "_conn_lock did not exist when _connect was called"


class TestConnectUsesLock:
    def test_connect_acquires_lock(self):
        """_connect() must hold _conn_lock while checking/setting self._conn."""
        base = _make_base()

        lock_held_during_connect = []
        original_connect = base._psycopg2.connect

        def patched_psycopg2_connect(dsn):
            # If the lock is held by _connect(), trying to acquire it here
            # (non-blocking) will fail.
            acquired = base._conn_lock.acquire(blocking=False)
            lock_held_during_connect.append(not acquired)
            if acquired:
                base._conn_lock.release()
            return original_connect(dsn)

        base._psycopg2.connect = patched_psycopg2_connect

        base._connect()

        assert lock_held_during_connect, "psycopg2.connect was never called"
        assert all(lock_held_during_connect), "_conn_lock was NOT held when psycopg2.connect ran"

    def test_connect_returns_same_conn_when_already_open(self):
        """When _conn is already open, _connect() returns it without re-connecting."""
        mock_conn = MagicMock()
        mock_conn.closed = False
        base = _make_base(mock_conn)
        base._conn = mock_conn

        result = base._connect()

        assert result is mock_conn
        base._psycopg2.connect.assert_not_called()

    def test_connect_reconnects_when_conn_is_none(self):
        """When _conn is None, _connect() creates a new connection."""
        base = _make_base()
        assert base._conn is None

        conn = base._connect()

        base._psycopg2.connect.assert_called_once_with(base._dsn)
        assert conn is not None

    def test_connect_reconnects_when_conn_is_closed(self):
        """When _conn.closed is truthy, _connect() creates a new connection."""
        mock_conn = MagicMock()
        mock_conn.closed = True
        base = _make_base(mock_conn)
        base._conn = mock_conn

        base._connect()

        base._psycopg2.connect.assert_called_once_with(base._dsn)

    def test_no_double_connect_under_concurrent_access(self):
        """Two threads calling _connect() simultaneously must produce exactly one connection."""
        call_count = 0
        connect_started = threading.Event()
        proceed = threading.Event()

        base = _make_base()

        original_connect = base._psycopg2.connect

        def slow_connect(dsn):
            nonlocal call_count
            call_count += 1
            connect_started.set()
            proceed.wait(timeout=5)  # hold the lock while the other thread is waiting
            return original_connect(dsn)

        base._psycopg2.connect = slow_connect

        results = []
        errors = []

        def worker():
            try:
                results.append(base._connect())
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        # Wait until t1 has started connecting (holding the lock), then start t2
        connect_started.wait(timeout=5)
        t2.start()
        # Let t1's slow_connect finish so both threads can complete
        proceed.set()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Threads raised: {errors}"
        # Only one real psycopg2.connect call must have happened
        assert call_count == 1, f"psycopg2.connect called {call_count} times — race condition!"
        assert results[0] is results[1], "Both threads must receive the same connection object"


class TestRollbackUsesLock:
    """Rollback-on-error paths must hold _conn_lock while touching self._conn."""

    def _assert_lock_held_during(self, base: _DataControllerBase, action) -> None:
        """Assert that _conn_lock is held (not acquirable) while `action` runs."""
        lock_held = []

        original_rollback = base._conn.rollback

        def patched_rollback():
            acquired = base._conn_lock.acquire(blocking=False)
            lock_held.append(not acquired)
            if acquired:
                base._conn_lock.release()
            return original_rollback()

        base._conn.rollback = patched_rollback
        action()

        assert lock_held, "rollback() was never called"
        assert all(lock_held), "_conn_lock was NOT held during rollback()"

    def test_annotation_write_label_rollback_holds_lock(self):
        from shared.data_controller.annotation import AnnotationDataController

        ctrl = object.__new__(AnnotationDataController)
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        ctrl._psycopg2 = mock_psycopg2
        ctrl._conn = mock_conn
        ctrl._conn_lock = threading.Lock()
        ctrl._dsn = "postgresql://test"

        # Make cursor.execute raise so the except block (with rollback) is reached
        mock_conn.cursor.return_value.__enter__.return_value.execute.side_effect = Exception(
            "db error"
        )

        def action():
            from uuid import uuid4

            try:
                ctrl.write_label(uuid4(), 5)
            except Exception:
                pass

        self._assert_lock_held_during(ctrl, action)

    def test_sampling_select_and_mark_rollback_holds_lock(self):
        from shared.data_controller.sampling import SamplingDataController

        ctrl = object.__new__(SamplingDataController)
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        ctrl._psycopg2 = mock_psycopg2
        ctrl._conn = mock_conn
        ctrl._conn_lock = threading.Lock()
        ctrl._dsn = "postgresql://test"

        mock_conn.cursor.return_value.__enter__.return_value.execute.side_effect = Exception(
            "db error"
        )

        def action():
            try:
                ctrl.select_and_mark_candidates(limit=10)
            except Exception:
                pass

        self._assert_lock_held_during(ctrl, action)

    def test_serving_store_prediction_rollback_holds_lock(self):
        from shared.data_controller.serving import ServingDataController

        ctrl = object.__new__(ServingDataController)
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        ctrl._psycopg2 = mock_psycopg2
        ctrl._conn = mock_conn
        ctrl._conn_lock = threading.Lock()
        ctrl._dsn = "postgresql://test"
        ctrl._available = True
        ctrl._failures = 0
        ctrl._s3_available = False

        mock_conn.cursor.return_value.__enter__.return_value.execute.side_effect = Exception(
            "db error"
        )

        from datetime import UTC, datetime
        from uuid import uuid4

        from shared.schemas.predict_record import PredictRecord

        record = PredictRecord(
            uuid=uuid4(),
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            model_version="v1",
            embedding=[0.0] * 64,
            prediction=0,
            confidence=0.9,
            prediction_distribution=[0.1] * 10,
            annotation_status="none",
            annotated_label=None,
        )

        def action():
            ctrl.store_prediction(record)

        self._assert_lock_held_during(ctrl, action)
