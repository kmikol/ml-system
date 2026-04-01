from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from shared.data_controller._base import DataControllerError
from shared.data_controller.dataset import DatasetController


class _FakeCursor:
    def __init__(
        self,
        *,
        fetchone_row=None,
        fetchall_rows=None,
        execute_error: Exception | None = None,
        rowcount: int = 0,
    ):
        self.fetchone_row = fetchone_row
        self.fetchall_rows = fetchall_rows or []
        self.execute_error = execute_error
        self.rowcount = rowcount
        self.executed: list[tuple[str, tuple | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        if self.execute_error is not None:
            raise self.execute_error
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_row

    def fetchall(self):
        return self.fetchall_rows


class _FakeConn:
    def __init__(self, cursors: list[_FakeCursor]):
        self._cursors = list(cursors)
        self.commit_calls = 0
        self.rollback_calls = 0

    def cursor(self):
        if not self._cursors:
            raise AssertionError("No cursor prepared")
        return self._cursors.pop(0)

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1


class _LakeFSFake:
    def __init__(self):
        self.put_calls: list[tuple[str, str, str, bytes]] = []
        self.commit_calls: list[tuple[str, str, str, dict[str, str]]] = []
        self.create_tag_calls: list[tuple[str, str, str]] = []
        self.ensure_repo_calls: list[tuple[str, str]] = []
        self.ensure_branch_calls: list[tuple[str, str]] = []
        self.resolve_ref_result: str | None = None

    def ensure_repo(self, repo: str, storage_namespace: str) -> None:
        self.ensure_repo_calls.append((repo, storage_namespace))

    def ensure_branch(self, repo: str, branch: str) -> None:
        self.ensure_branch_calls.append((repo, branch))

    def put_object(self, repo: str, branch: str, path: str, data: bytes) -> None:
        self.put_calls.append((repo, branch, path, data))

    def resolve_ref(self, repo: str, ref: str) -> str | None:
        return self.resolve_ref_result

    def commit(self, repo: str, branch: str, message: str, metadata: dict[str, str]) -> str:
        self.commit_calls.append((repo, branch, message, metadata))
        return "new-commit-id"

    def create_tag(self, repo: str, tag: str, ref: str) -> None:
        self.create_tag_calls.append((repo, tag, ref))


class _StoreFake:
    def __init__(self):
        self.put_calls: list[tuple[str, np.ndarray]] = []
        self.get_calls: list[str] = []

    def put_array(self, key: str, arr: np.ndarray) -> None:
        self.put_calls.append((key, arr))

    def get_array(self, key: str):
        self.get_calls.append(key)
        return np.array([[1.0]], dtype=np.float32)

    def get_array_or_none(self, key: str):
        self.get_calls.append(key)
        return None


def _base_controller(lakefs: _LakeFSFake, conn: _FakeConn) -> Any:
    ctrl: Any = cast(Any, DatasetController.__new__(DatasetController))
    ctrl._lakefs = lakefs
    ctrl._lakefs_repo = "repo"
    ctrl._lakefs_branch = "main"
    ctrl._conn = conn
    ctrl._connect = lambda: conn
    ctrl._ensure_lakefs_ready = lambda: None
    return ctrl


def test_ensure_lakefs_ready_raises_when_not_configured() -> None:
    ctrl: Any = cast(Any, DatasetController.__new__(DatasetController))
    ctrl._lakefs = None
    ctrl._lakefs_import_error = None
    ctrl._lakefs_ready = False

    with pytest.raises(DataControllerError, match="lakeFS is not configured"):
        ctrl._ensure_lakefs_ready()


def test_create_version_returns_existing_commit_when_already_registered() -> None:
    lakefs = _LakeFSFake()
    conn = _FakeConn([])
    ctrl = _base_controller(lakefs, conn)
    ctrl.get_version_info = lambda version_id: {"lakefs_commit_id": "existing-123"}

    commit_id = ctrl.create_version("v0", parent_version_id=None)

    assert commit_id == "existing-123"
    assert lakefs.put_calls == []
    assert lakefs.commit_calls == []
    assert lakefs.create_tag_calls == []


def test_create_version_raises_when_version_has_no_samples() -> None:
    lakefs = _LakeFSFake()
    samples_cursor = _FakeCursor(fetchall_rows=[])
    conn = _FakeConn([samples_cursor])
    ctrl = _base_controller(lakefs, conn)
    ctrl.get_version_info = lambda version_id: None

    with pytest.raises(DataControllerError, match="No samples found for version 'v-empty'"):
        ctrl.create_version("v-empty", parent_version_id=None)

    assert lakefs.put_calls == []
    assert conn.commit_calls == 0


def test_create_version_recovers_when_tag_already_exists() -> None:
    lakefs = _LakeFSFake()
    lakefs.resolve_ref_result = "existing-tag-commit"

    sample_rows = [
        (uuid4(), "train", 1, "obj/a.npy"),
        (uuid4(), "val", 2, "obj/b.npy"),
    ]
    conn = _FakeConn(
        [
            _FakeCursor(fetchall_rows=sample_rows),
            _FakeCursor(),
        ]
    )
    ctrl = _base_controller(lakefs, conn)
    ctrl.get_version_info = lambda version_id: None

    commit_id = ctrl.create_version("v1", parent_version_id="v0")

    assert commit_id == "existing-tag-commit"
    assert len(lakefs.put_calls) == 1
    assert lakefs.commit_calls == []
    assert lakefs.create_tag_calls == []
    assert conn.commit_calls == 1


def test_create_version_commits_and_tags_with_expected_metadata() -> None:
    lakefs = _LakeFSFake()
    sample_rows = [
        (uuid4(), "train", 1, "obj/a.npy"),
        (uuid4(), "train", 3, "obj/b.npy"),
        (uuid4(), "test", 9, "obj/c.npy"),
    ]
    conn = _FakeConn(
        [
            _FakeCursor(fetchall_rows=sample_rows),
            _FakeCursor(),
        ]
    )
    ctrl = _base_controller(lakefs, conn)
    ctrl.get_version_info = lambda version_id: None

    commit_id = ctrl.create_version("v2", parent_version_id="v1")

    assert commit_id == "new-commit-id"
    assert len(lakefs.put_calls) == 1
    repo, branch, path, manifest_bytes = lakefs.put_calls[0]
    assert (repo, branch, path) == ("repo", "main", "manifests/v2.json")

    manifest = json.loads(manifest_bytes.decode())
    assert manifest["version_id"] == "v2"
    assert manifest["parent_version_id"] == "v1"
    assert manifest["counts"] == {"train": 2, "test": 1}

    assert len(lakefs.commit_calls) == 1
    _, _, message, metadata = lakefs.commit_calls[0]
    assert message == "Dataset version v2"
    assert metadata == {
        "version_id": "v2",
        "parent_version_id": "v1",
        "sample_count": "3",
    }

    assert lakefs.create_tag_calls == [("repo", "dataset-v2", "new-commit-id")]
    assert conn.commit_calls == 1


def test_store_sample_uploads_float32_and_commits() -> None:
    store = _StoreFake()
    conn = _FakeConn([_FakeCursor()])

    ctrl: Any = cast(Any, DatasetController.__new__(DatasetController))
    ctrl._store = store
    ctrl._conn = conn
    ctrl._connect = lambda: conn

    sample_id = uuid4()
    ctrl.store_sample(
        uuid=sample_id,
        version_id="v0",
        split="train",
        label=5,
        image_2d=[[0.1, 0.2], [0.3, 0.4]],
        minio_path="20260401/sample.npy",
    )

    assert conn.commit_calls == 1
    assert len(store.put_calls) == 1
    key, arr = store.put_calls[0]
    assert key == "20260401/sample.npy"
    assert arr.dtype == np.float32
    assert arr.shape == (2, 2)


def test_get_dataset_split_downloads_images_for_rows() -> None:
    u1, u2 = uuid4(), uuid4()
    cur = _FakeCursor(fetchall_rows=[(u1, 1, "a.npy"), (u2, 2, "b.npy")])
    conn = _FakeConn([cur])
    store = _StoreFake()

    ctrl: Any = cast(Any, DatasetController.__new__(DatasetController))
    ctrl._store = store
    ctrl._connect = lambda: conn

    rows = ctrl.get_dataset_split("v0", "train")

    assert [r["uuid"] for r in rows] == [u1, u2]
    assert [r["label"] for r in rows] == [1, 2]
    assert store.get_calls == ["a.npy", "b.npy"]


def test_copy_version_returns_rowcount_and_commits() -> None:
    cur = _FakeCursor(rowcount=4)
    conn = _FakeConn([cur])

    ctrl: Any = cast(Any, DatasetController.__new__(DatasetController))
    ctrl._connect = lambda: conn

    inserted = ctrl.copy_version("v0", "v1")

    assert inserted == 4
    assert conn.commit_calls == 1
    assert cur.executed[0][1] == ("v1", "v0")


def test_get_unversioned_annotations_maps_rows() -> None:
    u1, u2 = uuid4(), uuid4()
    conn = _FakeConn([_FakeCursor(fetchall_rows=[(u1, 3), (u2, 8)])])

    ctrl: Any = cast(Any, DatasetController.__new__(DatasetController))
    ctrl._connect = lambda: conn

    out = ctrl.get_unversioned_annotations()

    assert out == [{"uuid": u1, "label": 3}, {"uuid": u2, "label": 8}]


def test_ensure_lakefs_ready_initializes_repo_branch_once(monkeypatch) -> None:
    lakefs = _LakeFSFake()
    ctrl: Any = cast(Any, DatasetController.__new__(DatasetController))
    ctrl._lakefs = lakefs
    ctrl._lakefs_ready = False

    monkeypatch.setenv("LAKEFS_REPO", "repo-x")
    monkeypatch.setenv("LAKEFS_BRANCH", "dev")
    monkeypatch.setenv("LAKEFS_STORAGE_NAMESPACE", "s3://custom-ns/")

    ctrl._ensure_lakefs_ready()

    assert ctrl._lakefs_ready is True
    assert lakefs.ensure_repo_calls == [("repo-x", "s3://custom-ns/")]
    assert lakefs.ensure_branch_calls == [("repo-x", "dev")]
    assert ctrl._lakefs_repo == "repo-x"
    assert ctrl._lakefs_branch == "dev"

    # Second call should be a no-op.
    ctrl._ensure_lakefs_ready()
    assert lakefs.ensure_repo_calls == [("repo-x", "s3://custom-ns/")]
    assert lakefs.ensure_branch_calls == [("repo-x", "dev")]


def test_init_fails_fast_when_lakefs_client_build_fails(monkeypatch) -> None:
    monkeypatch.setenv("DATA_CONTROLLER_DB_URL", "postgresql://ignored")
    monkeypatch.setenv("DATASET_S3_ENDPOINT_URL", "http://minio")
    monkeypatch.setenv("DATASET_BUCKET", "b")
    monkeypatch.setenv("LAKEFS_ENDPOINT_URL", "http://lakefs")

    fake_store = _StoreFake()

    with (
        patch("shared.data_controller.dataset._DataControllerBase.__init__", return_value=None),
        patch(
            "shared.data_controller.dataset.build_lakefs_client",
            side_effect=DataControllerError("lakefs sdk missing"),
        ),
        pytest.raises(DataControllerError, match="lakefs sdk missing"),
    ):
        DatasetController(object_store=fake_store)
