# shared/data_controller/_object_store.py
"""ObjectStore protocol and MinIO implementation for array storage.

Follows the same Protocol pattern used by model_artifact_controller/_protocol.py.
Swapping storage backends (e.g. MinIO → GCS) requires only a new class that
satisfies the ObjectStore protocol — nothing else in the codebase changes.
"""

from __future__ import annotations

import logging
from typing import Protocol

from shared.data_controller._base import DataControllerError

logger = logging.getLogger(__name__)


class ObjectStore(Protocol):
    """Interface for numpy array storage behind an S3-compatible backend.

    Write new backend implementations as classes that satisfy this Protocol.
    """

    def put_array(self, key: str, arr: "numpy.ndarray") -> None:
        """Serialize and upload a numpy array to the given key."""
        ...

    def get_array(self, key: str) -> "numpy.ndarray":
        """Download and deserialize a numpy array from the given key.

        Raises DataControllerError if the key does not exist.
        """
        ...

    def get_array_or_none(self, key: str) -> "numpy.ndarray | None":
        """Like get_array, but returns None if the key does not exist."""
        ...


class MinIOObjectStore:
    """S3-compatible object store backed by boto3.

    Lazily imports boto3 and numpy so they are only required by services
    that actually use the object store.
    """

    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        import boto3  # lazy

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._bucket = bucket

    def put_array(self, key: str, arr: "numpy.ndarray") -> None:
        import io

        import numpy as np

        buf = io.BytesIO()
        np.save(buf, np.array(arr, dtype=np.float32))
        buf.seek(0)
        try:
            self._client.upload_fileobj(buf, self._bucket, key)
        except Exception as exc:
            raise DataControllerError(f"Failed to upload '{key}': {exc}") from exc

    def get_array(self, key: str) -> "numpy.ndarray":
        import io

        import numpy as np

        buf = io.BytesIO()
        try:
            self._client.download_fileobj(self._bucket, key, buf)
        except Exception as exc:
            raise DataControllerError(f"Failed to download '{key}': {exc}") from exc
        buf.seek(0)
        return np.load(buf)

    def get_array_or_none(self, key: str) -> "numpy.ndarray | None":
        import io

        import numpy as np

        buf = io.BytesIO()
        try:
            self._client.download_fileobj(self._bucket, key, buf)
        except Exception as exc:
            # boto3 raises botocore.exceptions.ClientError with code
            # '404' or 'NoSuchKey' for missing objects.
            resp = getattr(exc, "response", None) or {}
            error_code = resp.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return None
            # For non-not-found errors, propagate as DataControllerError
            raise DataControllerError(f"Failed to download '{key}': {exc}") from exc
        buf.seek(0)
        return np.load(buf)
