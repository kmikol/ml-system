# shared/data_controller.py
"""
Data controller — the single point of contact between the rest of the system
and the operational data storage backend (currently Postgres).

Architecture:
  _DataControllerBase      — connection lifecycle, table creation
  ServingDataController    — store_prediction() with fire-and-forget error handling
  DriftDataController      — get_predictions(), get_labeled_predictions()
  SamplingDataController   — get_predictions(), mark_candidate(), count_labels_since()
  AnnotationDataController — write_label()
  FakeDataController       — in-memory implementation for unit tests

Services import their specific controller and instantiate it. Connection
management and error handling are invisible to callers.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from shared.config import require_env
from shared.schemas.predict_record import PredictRecord

logger = logging.getLogger(__name__)


class DataControllerError(Exception):
    """Raised when a data controller operation fails.

    No psycopg2 exception types escape this module — all failures are wrapped
    in this single type so callers write ``except DataControllerError``.
    """


# ── SQL ───────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id           TEXT        PRIMARY KEY,
    timestamp               TIMESTAMPTZ NOT NULL,
    model_version           TEXT        NOT NULL,
    image                   JSONB       NOT NULL,
    embedding               JSONB       NOT NULL,
    prediction              INTEGER     NOT NULL,
    confidence              REAL        NOT NULL,
    prediction_distribution JSONB       NOT NULL,
    label                   INTEGER,
    annotation_status       TEXT        NOT NULL DEFAULT 'none'
);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON predictions (timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version
    ON predictions (model_version);
"""

_INSERT = """
INSERT INTO predictions (
    prediction_id, timestamp, model_version, image, embedding,
    prediction, confidence, prediction_distribution, label, annotation_status
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (prediction_id) DO NOTHING;
"""

_SELECT_WINDOW = """
SELECT prediction_id, timestamp, model_version, image, embedding,
       prediction, confidence, prediction_distribution, label, annotation_status
FROM predictions
WHERE timestamp >= %s
  AND (%s IS NULL OR timestamp < %s)
  AND (%s IS NULL OR model_version = %s)
ORDER BY timestamp;
"""

_SELECT_LABELED = """
SELECT prediction_id, timestamp, model_version, image, embedding,
       prediction, confidence, prediction_distribution, label, annotation_status
FROM predictions
WHERE label IS NOT NULL AND timestamp >= %s
ORDER BY timestamp;
"""

_MARK_CANDIDATE = """
UPDATE predictions SET annotation_status = 'candidate'
WHERE prediction_id = %s AND annotation_status = 'none';
"""

_WRITE_LABEL = """
UPDATE predictions SET label = %s, annotation_status = 'annotated'
WHERE prediction_id = %s;
"""

_COUNT_LABELS = """
SELECT COUNT(*) FROM predictions
WHERE label IS NOT NULL AND timestamp >= %s;
"""


def _row_to_record(row: tuple) -> PredictRecord:
    (
        prediction_id, timestamp, model_version, image, embedding,
        prediction, confidence, prediction_distribution, label, annotation_status,
    ) = row
    return PredictRecord(
        prediction_id=prediction_id,
        timestamp=timestamp,
        model_version=model_version,
        image=image,
        embedding=embedding,
        prediction=prediction,
        confidence=confidence,
        prediction_distribution=prediction_distribution,
        label=label,
        annotation_status=annotation_status,
    )


# ── Base ──────────────────────────────────────────────────────────────────────

class _DataControllerBase:
    """Postgres connection lifecycle and table creation.

    All service-specific controllers inherit from this class. psycopg2 is
    imported lazily so services that don't use the data controller don't need
    it installed.
    """

    def __init__(self, dsn: str) -> None:
        import psycopg2  # lazy — keeps psycopg2 out of import-time for non-users
        self._psycopg2 = psycopg2
        self._conn = None
        self._dsn = dsn
        self._ensure_table()

    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg2.connect(self._dsn)
        return self._conn

    def _ensure_table(self) -> None:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
            conn.commit()
        except Exception as exc:
            raise DataControllerError(
                f"Failed to create predictions table: {exc}"
            ) from exc


# ── Service controllers ───────────────────────────────────────────────────────

class ServingDataController(_DataControllerBase):
    """Used by the serving service to persist prediction records.

    Graceful degradation: if Postgres is unavailable at startup, logs a warning
    and silently skips all writes. Write failures after connection are also
    swallowed — serving never raises due to DB issues.

    Instantiate at module level; no separate connect() call is needed.
    """

    def __init__(self) -> None:
        self._available = False
        self._failures = 0
        dsn = os.getenv("DATA_CONTROLLER_DB_URL", "")
        if not dsn:
            logger.warning(
                "DATA_CONTROLLER_DB_URL not set, predictions will not be persisted"
            )
            return
        try:
            super().__init__(dsn)
            self._available = True
            logger.info("Data controller connected")
        except Exception as exc:
            logger.warning(
                f"Data controller unavailable (serving continues without): {exc}"
            )

    def store_prediction(self, record: PredictRecord) -> None:
        """Persist a prediction record to Postgres.

        Fire-and-forget: if Postgres is unavailable or the write fails, the
        error is logged as a warning and swallowed. Serving never raises due
        to database issues.
        """
        if not self._available:
            return
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_INSERT, (
                    record.prediction_id,
                    record.timestamp,
                    record.model_version,
                    json.dumps(record.image),
                    json.dumps(record.embedding),
                    record.prediction,
                    record.confidence,
                    json.dumps(record.prediction_distribution),
                    record.label,
                    record.annotation_status,
                ))
            conn.commit()
        except Exception as exc:
            self._failures += 1
            logger.warning(f"Prediction storage failed ({self._failures} total): {exc}")
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None


class DriftDataController(_DataControllerBase):
    """Used by the drift monitoring service to query prediction history."""

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))

    def get_predictions(
        self,
        since: datetime,
        until: datetime | None = None,
        model_version: str | None = None,
    ) -> list[PredictRecord]:
        """Return predictions in [since, until), optionally filtered by model version."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_WINDOW, (since, until, until, model_version, model_version))
                return [_row_to_record(row) for row in cur.fetchall()]
        except Exception as exc:
            raise DataControllerError(f"Failed to query predictions: {exc}") from exc

    def get_labeled_predictions(self, since: datetime) -> list[PredictRecord]:
        """Return predictions that have a ground truth label, since *since*."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_LABELED, (since,))
                return [_row_to_record(row) for row in cur.fetchall()]
        except Exception as exc:
            raise DataControllerError(
                f"Failed to query labeled predictions: {exc}"
            ) from exc


class SamplingDataController(_DataControllerBase):
    """Used by the sample selection service."""

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))

    def get_predictions(
        self,
        since: datetime,
        until: datetime | None = None,
        model_version: str | None = None,
    ) -> list[PredictRecord]:
        """Return predictions in [since, until) for uncertainty/diversity scoring."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_WINDOW, (since, until, until, model_version, model_version))
                return [_row_to_record(row) for row in cur.fetchall()]
        except Exception as exc:
            raise DataControllerError(f"Failed to query predictions: {exc}") from exc

    def mark_candidate(self, prediction_id: str) -> None:
        """Advance annotation_status from 'none' to 'candidate'."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_MARK_CANDIDATE, (prediction_id,))
            conn.commit()
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(
                f"Failed to mark '{prediction_id}' as candidate: {exc}"
            ) from exc

    def count_labels_since(self, since: datetime) -> int:
        """Return the number of labeled predictions since *since*."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_COUNT_LABELS, (since,))
                return cur.fetchone()[0]
        except Exception as exc:
            raise DataControllerError(f"Failed to count labels: {exc}") from exc


class AnnotationDataController(_DataControllerBase):
    """Used by the annotation service to write ground truth labels."""

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))

    def write_label(self, prediction_id: str, label: int) -> None:
        """Write a ground truth label and advance annotation_status to 'annotated'."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_WRITE_LABEL, (label, prediction_id))
            conn.commit()
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(
                f"Failed to write label for '{prediction_id}': {exc}"
            ) from exc


# ── Dataset controller ────────────────────────────────────────────────────────

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
        import io

        import numpy as np

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


# ── Fake (for tests) ──────────────────────────────────────────────────────────

class FakeDataController:
    """In-memory implementation for unit tests — no Postgres required.

    Implements the full surface area of all service controllers. Use in place
    of any service-specific controller by duck typing.
    """

    def __init__(self) -> None:
        self._records: list[PredictRecord] = []

    def store_prediction(self, record: PredictRecord) -> None:
        self._records.append(record.model_copy())

    def get_predictions(
        self,
        since: datetime,
        until: datetime | None = None,
        model_version: str | None = None,
    ) -> list[PredictRecord]:
        result = [r for r in self._records if r.timestamp >= since]
        if until is not None:
            result = [r for r in result if r.timestamp < until]
        if model_version is not None:
            result = [r for r in result if r.model_version == model_version]
        return result

    def get_labeled_predictions(self, since: datetime) -> list[PredictRecord]:
        return [r for r in self._records if r.label is not None and r.timestamp >= since]

    def mark_candidate(self, prediction_id: str) -> None:
        for r in self._records:
            if r.prediction_id == prediction_id and r.annotation_status == "none":
                r.annotation_status = "candidate"
                return

    def write_label(self, prediction_id: str, label: int) -> None:
        for r in self._records:
            if r.prediction_id == prediction_id:
                r.label = label
                r.annotation_status = "annotated"
                return

    def count_labels_since(self, since: datetime) -> int:
        return sum(
            1 for r in self._records
            if r.label is not None and r.timestamp >= since
        )
