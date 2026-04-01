# shared/data_controller/dataset.py
"""DatasetController — versioned dataset management in Postgres (metadata) and object storage (images)."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from shared.config import require_env
from shared.data_controller._base import DataControllerError, _DataControllerBase

if TYPE_CHECKING:
    from shared.data_controller._lakefs import LakeFSClient
    from shared.data_controller._object_store import ObjectStore

logger = logging.getLogger(__name__)

# ── SQL ───────────────────────────────────────────────────────────────────────

_INSERT_SAMPLE = """
INSERT INTO dataset_samples (uuid, version_id, split, label, minio_path)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (uuid, version_id) DO NOTHING;
"""

_SELECT_SPLIT = """
SELECT uuid, label, minio_path
FROM dataset_samples
WHERE version_id = %s AND split = %s
ORDER BY uuid;
"""

# Return the version_id whose most-recently-inserted row has the latest
# created_at timestamp — i.e. the last version seeded.
_SELECT_LATEST_VERSION = """
SELECT version_id
FROM dataset_samples
GROUP BY version_id
ORDER BY MAX(created_at) DESC
LIMIT 1;
"""

_SELECT_UNVERSIONED_ANNOTATIONS = """
SELECT p.uuid, p.annotated_label
FROM predictions p
WHERE p.annotation_status = 'annotated'
  AND p.uuid NOT IN (SELECT uuid FROM dataset_samples);
"""

_COPY_VERSION = """
INSERT INTO dataset_samples (uuid, version_id, split, label, minio_path)
SELECT uuid, %s, split, label, minio_path
FROM dataset_samples
WHERE version_id = %s
ON CONFLICT (uuid, version_id) DO NOTHING;
"""

# ── SQL — dataset_versions table ──────────────────────────────────────────────

_INSERT_VERSION = """
INSERT INTO dataset_versions (version_id, parent_version_id, lakefs_commit_id, lakefs_tag, sample_count)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (version_id) DO NOTHING;
"""

_SELECT_VERSION = """
SELECT version_id, parent_version_id, lakefs_commit_id, lakefs_tag, sample_count, created_at
FROM dataset_versions
WHERE version_id = %s;
"""

_SELECT_VERSION_HISTORY = """
SELECT version_id, parent_version_id, lakefs_commit_id, lakefs_tag, sample_count, created_at
FROM dataset_versions
ORDER BY created_at;
"""

_SELECT_ALL_SAMPLES_FOR_VERSION = """
SELECT uuid, split, label, minio_path
FROM dataset_samples
WHERE version_id = %s
ORDER BY uuid;
"""

_COUNT_SAMPLES_FOR_VERSION = """
SELECT COUNT(*)
FROM dataset_samples
WHERE version_id = %s;
"""


def _build_default_object_store() -> ObjectStore:
    """Build a MinIOObjectStore from environment variables."""
    from shared.data_controller._object_store import MinIOObjectStore

    return MinIOObjectStore(
        endpoint_url=require_env("DATASET_S3_ENDPOINT_URL"),
        bucket=require_env("DATASET_BUCKET"),
        access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    )


class DatasetController(_DataControllerBase):
    """Manages versioned dataset samples stored in Postgres (metadata) and object storage (images).

    Dataset versions are tracked in two layers:
      - ``dataset_samples`` (Postgres): fast operational queries by version+split.
      - lakeFS: immutable commit snapshots with manifests for reproducibility.

    Args:
        object_store: Optional ``ObjectStore`` implementation. If ``None``,
            a ``MinIOObjectStore`` is built from environment variables.
        lakefs: Optional ``LakeFSClient``. If ``None`` and ``LAKEFS_ENDPOINT_URL``
            is set, a client is built from environment variables. If lakeFS env
            vars are absent, versioning methods will raise ``DataControllerError``
            while all other methods (e.g. ``get_dataset_split``) work normally.
    """

    def __init__(
        self,
        object_store: ObjectStore | None = None,
        lakefs: LakeFSClient | None = None,
    ) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))
        self._store: ObjectStore = object_store or _build_default_object_store()

        # lakeFS is optional: only built when LAKEFS_ENDPOINT_URL is present.
        # Repo/branch setup is deferred to the first versioning call.
        if lakefs is not None:
            self._lakefs: LakeFSClient | None = lakefs
        elif os.environ.get("LAKEFS_ENDPOINT_URL"):
            from shared.data_controller._lakefs import build_lakefs_client

            self._lakefs = build_lakefs_client()
        else:
            self._lakefs = None

        self._lakefs_ready = False  # repo/branch setup deferred until needed

    def _ensure_lakefs_ready(self) -> None:
        """Lazy-initialize lakeFS repo and branch on first versioning call.

        Raises:
            DataControllerError: If lakeFS env vars are not configured.
        """
        if self._lakefs is None:
            raise DataControllerError(
                "lakeFS is not configured. Set LAKEFS_ENDPOINT_URL, "
                "LAKEFS_ACCESS_KEY_ID, and LAKEFS_SECRET_ACCESS_KEY to enable versioning."
            )
        if self._lakefs_ready:
            return
        self._lakefs_repo = require_env("LAKEFS_REPO")
        self._lakefs_branch = os.environ.get("LAKEFS_BRANCH", "main")
        storage_ns = os.environ.get(
            "LAKEFS_STORAGE_NAMESPACE",
            f"s3://{os.environ.get('DATASET_BUCKET', 'lakefs-data')}/",
        )
        self._lakefs.ensure_repo(self._lakefs_repo, storage_ns)
        self._lakefs.ensure_branch(self._lakefs_repo, self._lakefs_branch)
        self._lakefs_ready = True

    # ── Sample management ─────────────────────────────────────────────────────

    def store_sample(
        self,
        uuid: UUID,
        version_id: str,
        split: str,
        label: int,
        image_2d: list,
        minio_path: str,
    ) -> None:
        """Upload image to object storage and upsert the sample row into ``dataset_samples``.

        Idempotent: ON CONFLICT (uuid, version_id) DO NOTHING — re-seeding the
        same sample into the same version is safe.

        Args:
            uuid: Stable UUID for this sample (assigned at data-preparation time).
            version_id: Dataset version this sample belongs to (e.g. ``'v0'``).
            split: One of ``'train'``, ``'val'``, ``'test'``.
            label: Ground truth class label (0–9).
            image_2d: 14×14 float32 pixel values in [0, 1].
            minio_path: Key within the bucket (e.g. ``'20260322/{uuid}.npy'``).
        """
        import numpy as np

        self._store.put_array(minio_path, np.array(image_2d, dtype=np.float32))

        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_INSERT_SAMPLE, (uuid, version_id, split, label, minio_path))
            conn.commit()
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(f"Failed to store sample '{uuid}': {exc}") from exc

    # ── Data retrieval ────────────────────────────────────────────────────────

    def get_latest_version(self) -> str | None:
        """Return the most recently seeded dataset version ID, or ``None`` if none exist."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_LATEST_VERSION)
                row = cur.fetchone()
            return row[0] if row else None
        except Exception as exc:
            raise DataControllerError(f"Failed to query latest version: {exc}") from exc

    def get_dataset_split(self, version_id: str, split: str) -> list[dict]:
        """Fetch all samples for a version+split, loading images from object storage.

        Args:
            version_id: Dataset version to query (e.g. ``'v0'``).
            split: One of ``'train'``, ``'val'``, ``'test'``.

        Returns:
            List of dicts with keys: ``uuid``, ``label``,
            ``image`` (14×14 ndarray), ``minio_path``.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_SPLIT, (version_id, split))
                rows = cur.fetchall()
        except Exception as exc:
            raise DataControllerError(
                f"Failed to query split '{split}' for version '{version_id}': {exc}"
            ) from exc

        return [
            {
                "uuid": uuid,
                "label": label,
                "image": self.download_image(minio_path),
                "minio_path": minio_path,
            }
            for uuid, label, minio_path in rows
        ]

    def get_unversioned_annotations(self) -> list[dict]:
        """Return annotated predictions not yet included in any dataset version.

        Returns:
            List of dicts with keys: ``uuid`` (UUID), ``label`` (int).
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_UNVERSIONED_ANNOTATIONS)
                rows = cur.fetchall()
            return [{"uuid": uuid, "label": label} for uuid, label in rows]
        except Exception as exc:
            raise DataControllerError(f"Failed to query unversioned annotations: {exc}") from exc

    def copy_version(self, src_version_id: str, dst_version_id: str) -> int:
        """Copy all samples from ``src_version_id`` into ``dst_version_id``.

        Pure SQL — no object storage operations. The ``minio_path`` values are preserved
        as-is so the new version points to the same objects. Idempotent via
        ON CONFLICT DO NOTHING.

        Args:
            src_version_id: Existing version to copy from (e.g. ``'v0'``).
            dst_version_id: New version to copy into (e.g. ``'v1'``).

        Returns:
            Number of rows inserted (0 if already fully copied).
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_COPY_VERSION, (dst_version_id, src_version_id))
                count = cur.rowcount
            conn.commit()
            return count
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(
                f"Failed to copy version '{src_version_id}' → '{dst_version_id}': {exc}"
            ) from exc

    def download_image(self, minio_path: str):
        """Download and deserialize a single image .npy file from object storage."""
        return self._store.get_array(minio_path)

    def download_image_or_none(self, minio_path: str):
        """Download an image, returning ``None`` if the key does not exist."""
        return self._store.get_array_or_none(minio_path)

    # ── Version management (lakeFS) ──────────────────────────────────────────

    def create_version(self, version_id: str, parent_version_id: str | None) -> str:
        """Register a dataset version in Postgres and commit a manifest to lakeFS.

        Idempotent: if ``version_id`` is already registered in ``dataset_versions``,
        the existing ``lakefs_commit_id`` is returned immediately without creating a
        new commit or tag.

        Steps (first call only):
          1. Query ``dataset_samples`` for this version_id and build a manifest.
          2. Upload the manifest to lakeFS on the configured branch.
          3. Commit and create an immutable tag ``dataset-{version_id}``.
          4. Insert a row into ``dataset_versions``.

        Args:
            version_id: The version to register (e.g. ``'v0'``, ``'v1'``).
            parent_version_id: The previous version this was derived from,
                or ``None`` for the initial version.

        Returns:
            The lakeFS commit ID.

        Raises:
            DataControllerError: If lakeFS is not configured, if no samples
                exist for the version, or if any step fails.
        """
        self._ensure_lakefs_ready()
        assert self._lakefs is not None  # guaranteed by _ensure_lakefs_ready()

        # Idempotency check: if this version is already registered, return the
        # recorded commit ID so callers always get the canonical, tag-backed value.
        existing = self.get_version_info(version_id)
        if existing is not None:
            logger.info(
                f"Dataset version '{version_id}' already registered; "
                f"returning existing commit {existing['lakefs_commit_id']}"
            )
            return existing["lakefs_commit_id"]

        # 1. Build manifest from dataset_samples
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_ALL_SAMPLES_FOR_VERSION, (version_id,))
                rows = cur.fetchall()
        except Exception as exc:
            raise DataControllerError(
                f"Failed to query samples for version '{version_id}': {exc}"
            ) from exc

        if not rows:
            raise DataControllerError(
                f"No samples found for version '{version_id}'. "
                "Store samples before calling create_version()."
            )

        samples = [
            {
                "uuid": str(uuid),
                "split": split,
                "label": label,
                "object_key": minio_path,
            }
            for uuid, split, label, minio_path in rows
        ]

        counts: dict[str, int] = {}
        for s in samples:
            counts[s["split"]] = counts.get(s["split"], 0) + 1

        manifest = {
            "version_id": version_id,
            "parent_version_id": parent_version_id,
            "created_at": datetime.now(UTC).isoformat(),
            "samples": samples,
            "counts": counts,
        }

        manifest_path = f"manifests/{version_id}.json"
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode()
        tag_name = f"dataset-{version_id}"

        # 2. Upload manifest to lakeFS
        self._lakefs.put_object(
            self._lakefs_repo,
            self._lakefs_branch,
            manifest_path,
            manifest_bytes,
        )

        # 3. Commit and tag — tolerate partial-failure recovery where a prior
        #    run created the tag but did not insert the DB row.
        existing_commit = self._lakefs.resolve_ref(self._lakefs_repo, tag_name)
        if existing_commit is not None:
            # Tag already exists (e.g. prior run crashed after lakeFS write but
            # before Postgres insert).  Re-use the canonical commit rather than
            # creating a new one.
            commit_id = existing_commit
            logger.warning(
                f"Tag '{tag_name}' already exists on lakeFS (commit {commit_id}); "
                "recovering missing database row."
            )
        else:
            commit_id = self._lakefs.commit(
                self._lakefs_repo,
                self._lakefs_branch,
                message=f"Dataset version {version_id}",
                metadata={
                    "version_id": version_id,
                    "parent_version_id": parent_version_id or "",
                    "sample_count": str(len(samples)),
                },
            )
            self._lakefs.create_tag(self._lakefs_repo, tag_name, commit_id)

        # 4. Register in Postgres
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_VERSION,
                    (version_id, parent_version_id, commit_id, tag_name, len(samples)),
                )
            conn.commit()
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(
                f"Failed to register version '{version_id}' in database: {exc}"
            ) from exc

        logger.info(
            f"Created dataset version '{version_id}': "
            f"{len(samples)} samples, lakeFS commit {commit_id}, tag {tag_name}"
        )
        return commit_id

    def get_version_info(self, version_id: str) -> dict | None:
        """Return metadata for a dataset version, or ``None`` if not found.

        Returns:
            Dict with keys: ``version_id``, ``parent_version_id``,
            ``lakefs_commit_id``, ``lakefs_tag``, ``sample_count``, ``created_at``.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_VERSION, (version_id,))
                row = cur.fetchone()
            if row is None:
                return None
            return {
                "version_id": row[0],
                "parent_version_id": row[1],
                "lakefs_commit_id": row[2],
                "lakefs_tag": row[3],
                "sample_count": row[4],
                "created_at": row[5],
            }
        except Exception as exc:
            raise DataControllerError(
                f"Failed to query version info for '{version_id}': {exc}"
            ) from exc

    def get_version_history(self) -> list[dict]:
        """Return all dataset versions ordered by creation time.

        Returns:
            List of dicts, each with the same keys as ``get_version_info()``.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_VERSION_HISTORY)
                rows = cur.fetchall()
            return [
                {
                    "version_id": row[0],
                    "parent_version_id": row[1],
                    "lakefs_commit_id": row[2],
                    "lakefs_tag": row[3],
                    "sample_count": row[4],
                    "created_at": row[5],
                }
                for row in rows
            ]
        except Exception as exc:
            raise DataControllerError(f"Failed to query version history: {exc}") from exc
