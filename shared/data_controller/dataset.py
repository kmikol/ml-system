# shared/data_controller/dataset.py
"""DatasetController — versioned dataset management in Postgres (metadata) and MinIO (images)."""

from __future__ import annotations

import os
from uuid import UUID

from shared.config import require_env
from shared.data_controller._base import DataControllerError, _DataControllerBase

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


class DatasetController(_DataControllerBase):
    """Manages versioned dataset samples stored in Postgres (metadata) and MinIO (images).

    Schema:
      - ``dataset_samples``: membership table — which sample UUIDs belong to each
        dataset version/split, together with their label and MinIO path.

    Postgres holds metadata and MinIO holds the actual image bytes (.npy files).
    """

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))
        import boto3  # lazy — only needed by the dataset controller

        self._s3 = boto3.client(
            "s3",
            endpoint_url=require_env("DATASET_S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )
        self._bucket = require_env("DATASET_BUCKET")

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
        """Upload image to MinIO and upsert the sample row into ``dataset_samples``.

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
        import io

        import numpy as np

        buf = io.BytesIO()
        np.save(buf, np.array(image_2d, dtype=np.float32))
        buf.seek(0)
        self._s3.upload_fileobj(buf, self._bucket, minio_path)

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
        """Fetch all samples for a version+split, loading images from MinIO.

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

    def download_image(self, minio_path: str):
        """Download and deserialize a single image .npy file from MinIO."""
        import io

        import numpy as np

        buf = io.BytesIO()
        self._s3.download_fileobj(self._bucket, minio_path, buf)
        buf.seek(0)
        return np.load(buf)
