# shared/data_controller/_lakefs.py
"""LakeFSClient — thin wrapper around the lakeFS Python SDK.

All lakeFS-specific details are encapsulated here.  The rest of the codebase
interacts with lakeFS only through this module and never imports ``lakefs``
directly.  All errors are wrapped in ``DataControllerError``.
"""

from __future__ import annotations

import logging
import os

from shared.config import require_env
from shared.data_controller._base import DataControllerError

logger = logging.getLogger(__name__)


class LakeFSClient:
    """Wrapper around the ``lakefs`` high-level Python SDK.

    Lazily imports ``lakefs`` so services that don't need lakeFS don't
    require it installed at import time.

    Args:
        endpoint_url: lakeFS API endpoint (e.g. ``http://lakefs:8000``).
            ``/api/v1`` is appended automatically if not present.
        access_key: lakeFS access key ID.
        secret_key: lakeFS secret access key.
    """

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str) -> None:
        import lakefs  # lazy

        host = endpoint_url.rstrip("/")
        if not host.endswith("/api/v1"):
            host = f"{host}/api/v1"

        self._client = lakefs.Client(
            host=host,
            username=access_key,
            password=secret_key,
        )
        self._lakefs = lakefs

    # ── Repository ────────────────────────────────────────────────────────────

    def ensure_repo(self, repo: str, storage_namespace: str) -> None:
        """Create the repository if it does not already exist."""
        try:
            self._lakefs.Repository(repo, client=self._client).create(
                storage_namespace=storage_namespace,
                default_branch="main",
                exist_ok=True,
            )
        except Exception as exc:
            raise DataControllerError(f"Failed to ensure lakeFS repo '{repo}': {exc}") from exc

    # ── Branch ────────────────────────────────────────────────────────────────

    def ensure_branch(self, repo: str, branch: str, source: str = "main") -> None:
        """Create a branch if it does not already exist."""
        try:
            r = self._lakefs.Repository(repo, client=self._client)
            r.branch(branch).create(source_reference=source, exist_ok=True)
        except Exception as exc:
            raise DataControllerError(
                f"Failed to ensure branch '{branch}' on repo '{repo}': {exc}"
            ) from exc

    # ── Object I/O ────────────────────────────────────────────────────────────

    def put_object(self, repo: str, branch: str, path: str, data: bytes) -> None:
        """Upload raw bytes to a path on a branch."""
        try:
            r = self._lakefs.Repository(repo, client=self._client)
            r.branch(branch).object(path).upload(data=data, mode="wb")
        except Exception as exc:
            raise DataControllerError(
                f"Failed to upload '{path}' to '{repo}/{branch}': {exc}"
            ) from exc

    def get_object(self, repo: str, ref: str, path: str) -> bytes:
        """Download raw bytes from a path at a ref (branch, tag, or commit ID)."""
        try:
            r = self._lakefs.Repository(repo, client=self._client)
            with r.ref(ref).object(path).reader(mode="rb") as reader:
                return reader.read()
        except Exception as exc:
            raise DataControllerError(
                f"Failed to download '{path}' from '{repo}/{ref}': {exc}"
            ) from exc

    # ── Commit / Tag ──────────────────────────────────────────────────────────

    def commit(
        self,
        repo: str,
        branch: str,
        message: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Commit staged changes on a branch.

        Returns:
            The commit ID of the new commit.
        """
        try:
            r = self._lakefs.Repository(repo, client=self._client)
            ref = r.branch(branch).commit(message=message, metadata=metadata or {})
            return ref.get_commit().id
        except Exception as exc:
            raise DataControllerError(
                f"Failed to commit on '{repo}/{branch}': {exc}"
            ) from exc

    def create_tag(self, repo: str, tag: str, ref: str) -> None:
        """Create an immutable tag pointing at a ref.

        Idempotent: no-op if the tag already exists (``exist_ok=True``).
        """
        try:
            r = self._lakefs.Repository(repo, client=self._client)
            r.tag(tag).create(source_ref=ref, exist_ok=True)
        except Exception as exc:
            raise DataControllerError(
                f"Failed to create tag '{tag}' on repo '{repo}': {exc}"
            ) from exc


def build_lakefs_client() -> LakeFSClient:
    """Build a ``LakeFSClient`` from environment variables.

    Required env vars:
      - ``LAKEFS_ENDPOINT_URL``
      - ``LAKEFS_ACCESS_KEY_ID``
      - ``LAKEFS_SECRET_ACCESS_KEY``
    """
    return LakeFSClient(
        endpoint_url=require_env("LAKEFS_ENDPOINT_URL"),
        access_key=require_env("LAKEFS_ACCESS_KEY_ID"),
        secret_key=require_env("LAKEFS_SECRET_ACCESS_KEY"),
    )
