from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import shared.data_controller._lakefs as lakefs_module
import shared.data_controller._object_store as object_store_module
from shared.data_controller._base import DataControllerError
from shared.data_controller._lakefs import LakeFSClient
from shared.data_controller._object_store import MinIOObjectStore


class _DownloadError(Exception):
    def __init__(self, code: str):
        super().__init__(f"download failed: {code}")
        self.response = {"Error": {"Code": code}}


def test_object_store_get_array_or_none_returns_none_for_missing_key() -> None:
    class _Client:
        def download_fileobj(self, bucket, key, buf):
            raise _DownloadError("NoSuchKey")

    store = MinIOObjectStore.__new__(MinIOObjectStore)
    store._client = _Client()
    store._bucket = "b"

    assert store.get_array_or_none("missing.npy") is None


def test_object_store_get_array_or_none_raises_for_non_404_errors() -> None:
    class _Client:
        def download_fileobj(self, bucket, key, buf):
            raise _DownloadError("AccessDenied")

    store = MinIOObjectStore.__new__(MinIOObjectStore)
    store._client = _Client()
    store._bucket = "b"

    with pytest.raises(DataControllerError, match="Failed to download 'x.npy'"):
        store.get_array_or_none("x.npy")


def test_object_store_put_array_coerces_to_float32_before_upload() -> None:
    uploaded = {}

    class _Client:
        def upload_fileobj(self, fileobj, bucket, key):
            uploaded["bucket"] = bucket
            uploaded["key"] = key
            uploaded["bytes"] = fileobj.read()

    store = MinIOObjectStore.__new__(MinIOObjectStore)
    store._client = _Client()
    store._bucket = "bucket-a"

    src = np.array([[1.0, 2.0]], dtype=np.float64)
    store.put_array("arr.npy", src)

    arr = np.load(io.BytesIO(uploaded["bytes"]))
    assert uploaded["bucket"] == "bucket-a"
    assert uploaded["key"] == "arr.npy"
    assert arr.dtype == np.float32


def test_object_store_put_array_wraps_upload_errors() -> None:
    class _Client:
        def upload_fileobj(self, fileobj, bucket, key):
            raise RuntimeError("upload failed")

    store = MinIOObjectStore.__new__(MinIOObjectStore)
    store._client = _Client()
    store._bucket = "bucket-a"

    with pytest.raises(DataControllerError, match="Failed to upload 'arr.npy'"):
        store.put_array("arr.npy", np.array([[1.0]], dtype=np.float32))


def test_object_store_get_array_downloads_and_loads_numpy() -> None:
    payload = io.BytesIO()
    np.save(payload, np.array([[7.0]], dtype=np.float32))
    raw = payload.getvalue()

    class _Client:
        def download_fileobj(self, bucket, key, buf):
            buf.write(raw)

    store = MinIOObjectStore.__new__(MinIOObjectStore)
    store._client = _Client()
    store._bucket = "b"

    arr = store.get_array("x.npy")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 1)
    assert arr.dtype == np.float32


def test_object_store_get_array_wraps_download_errors() -> None:
    class _Client:
        def download_fileobj(self, bucket, key, buf):
            raise RuntimeError("network")

    store = MinIOObjectStore.__new__(MinIOObjectStore)
    store._client = _Client()
    store._bucket = "b"

    with pytest.raises(DataControllerError, match="Failed to download 'x.npy'"):
        store.get_array("x.npy")


def test_minio_object_store_init_builds_boto3_client(monkeypatch) -> None:
    captured = {}

    class _Boto3:
        def client(self, service, endpoint_url, aws_access_key_id, aws_secret_access_key):
            captured["service"] = service
            captured["endpoint_url"] = endpoint_url
            captured["aws_access_key_id"] = aws_access_key_id
            captured["aws_secret_access_key"] = aws_secret_access_key
            return object()

    monkeypatch.setattr(object_store_module, "boto3", _Boto3())

    store = MinIOObjectStore(
        endpoint_url="http://minio:9000",
        bucket="mnist",
        access_key="ak",
        secret_key="sk",
    )

    assert store._bucket == "mnist"
    assert captured == {
        "service": "s3",
        "endpoint_url": "http://minio:9000",
        "aws_access_key_id": "ak",
        "aws_secret_access_key": "sk",
    }


def test_lakefs_init_appends_api_v1_suffix(monkeypatch) -> None:
    captured = {}

    class _FakeClient:
        def __init__(self, host, username, password):
            captured["host"] = host
            captured["username"] = username
            captured["password"] = password

    fake_module = SimpleNamespace(Client=_FakeClient)
    monkeypatch.setattr(lakefs_module, "lakefs", fake_module)

    LakeFSClient(endpoint_url="http://lakefs:8000", access_key="ak", secret_key="sk")

    assert captured["host"] == "http://lakefs:8000/api/v1"
    assert captured["username"] == "ak"
    assert captured["password"] == "sk"


def test_lakefs_commit_uses_empty_metadata_when_none() -> None:
    captured = {}

    class _CommitRef:
        def get_commit(self):
            return SimpleNamespace(id="commit-123")

    class _Branch:
        def commit(self, message, metadata):
            captured["message"] = message
            captured["metadata"] = metadata
            return _CommitRef()

    class _Repo:
        def __init__(self, name, client):
            self.name = name
            self.client = client

        def branch(self, branch):
            captured["repo"] = self.name
            captured["branch"] = branch
            return _Branch()

    client: Any = cast(Any, LakeFSClient.__new__(LakeFSClient))
    client._client = object()
    client._lakefs = SimpleNamespace(Repository=_Repo)

    commit_id = client.commit(repo="r1", branch="main", message="m1", metadata=None)

    assert commit_id == "commit-123"
    assert captured["repo"] == "r1"
    assert captured["branch"] == "main"
    assert captured["message"] == "m1"
    assert captured["metadata"] == {}
