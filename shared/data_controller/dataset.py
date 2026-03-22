# shared/data_controller/dataset.py
"""DatasetController — versioned dataset samples in Postgres (metadata) and MinIO (images)."""

from __future__ import annotations

import os

from shared.config import require_env
from shared.data_controller._base import DataControllerError, _DataControllerBase

# ── SQL — dataset_samples table ───────────────────────────────────────────────

_CREATE_DATASET_TABLE = """
CREATE TABLE IF NOT EXISTS dataset_samples (
    sample_id  TEXT    PRIMARY KEY,
    split      TEXT    NOT NULL,
    label      INTEGER NOT NULL,
    minio_path TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dataset_samples_split
    ON dataset_samples (split);
"""

_INSERT_DATASET_SAMPLE = """
INSERT INTO dataset_samples (sample_id, split, label, minio_path)
VALUES (%s, %s, %s, %s)
ON CONFLICT (sample_id) DO NOTHING;
"""

_SELECT_DATASET_SPLIT = """
SELECT sample_id, label, minio_path
FROM dataset_samples
WHERE split = %s
ORDER BY sample_id;
"""


class DatasetController(_DataControllerBase):
    """Manages versioned dataset samples stored in Postgres (metadata) and MinIO (images).

    Postgres is the index: it maps sample_id → split, label, minio_path.
    MinIO holds the actual image bytes (float32 .npy files).
    """

    def _ensure_table(self) -> None:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_CREATE_DATASET_TABLE)
            conn.commit()
        except Exception as exc:
            raise DataControllerError(
                f"Failed to create dataset_samples table: {exc}"
            ) from exc

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))
        import boto3  # lazy — only needed by dataset controller

        self._s3 = boto3.client(
            "s3",
            endpoint_url=require_env("DATASET_S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )
        self._bucket = require_env("DATASET_BUCKET")

    def store_sample(
        self, sample_id: str, split: str, label: int, image_2d: list, minio_path: str
    ) -> None:
        """Upload image to MinIO and insert metadata row into Postgres.

        Args:
            sample_id: Unique identifier for this sample (UUID).
            split: Dataset split ('train', 'val', 'test').
            label: Ground truth class label (0–9).
            image_2d: 14×14 float32 pixel values in [0, 1].
            minio_path: Key within the bucket (e.g. '20260322/{uuid}.npy').
        """
        import io

        import numpy as np

        # Upload to MinIO
        buf = io.BytesIO()
        np.save(buf, np.array(image_2d, dtype=np.float32))
        buf.seek(0)
        self._s3.upload_fileobj(buf, self._bucket, minio_path)

        # Insert metadata
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_INSERT_DATASET_SAMPLE, (sample_id, split, label, minio_path))
            conn.commit()
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(f"Failed to store sample '{sample_id}': {exc}") from exc

    def get_dataset_split(self, split: str) -> list[dict]:
        """Fetch all samples for a split — queries Postgres for paths, loads images from MinIO.

        Returns:
            List of dicts with keys: sample_id, label, image (14×14 ndarray), minio_path.
        """


        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_DATASET_SPLIT, (split,))
                rows = cur.fetchall()
        except Exception as exc:
            raise DataControllerError(f"Failed to query split '{split}': {exc}") from exc

        samples = []
        for sample_id, label, minio_path in rows:
            image = self.download_image(minio_path)
            samples.append({"sample_id": sample_id, "label": label, "image": image, "minio_path": minio_path})
        return samples

    def download_image(self, minio_path: str):
        """Download and deserialize a single image .npy from MinIO."""
        import io

        import numpy as np

        buf = io.BytesIO()
        self._s3.download_fileobj(self._bucket, minio_path, buf)
        buf.seek(0)
        return np.load(buf)
