# shared/data_controller/_base.py
"""
Base class, shared SQL constants, and error type for all data controllers.
"""

from __future__ import annotations

import logging

from shared.schemas.predict_record import PredictRecord

logger = logging.getLogger(__name__)


class DataControllerError(Exception):
    """Raised when a data controller operation fails.

    No psycopg2 exception types escape this module — all failures are wrapped
    in this single type so callers write ``except DataControllerError``.
    """


# ── SQL — predictions table ───────────────────────────────────────────────────

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

_MARK_CANDIDATES_BATCH = """
UPDATE predictions
SET annotation_status = 'candidate'
WHERE prediction_id IN (
    SELECT p.prediction_id
    FROM predictions p
    JOIN dataset_samples d ON d.sample_id = p.prediction_id
    WHERE p.annotation_status = 'none'
    ORDER BY RANDOM()
    LIMIT %s
)
RETURNING prediction_id;
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
