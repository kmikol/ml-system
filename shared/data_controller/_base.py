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


# ── Schema DDL ────────────────────────────────────────────────────────────────
#
# Both tables are created in a single transaction on controller startup.
# gen_random_uuid() requires the pgcrypto extension; CREATE EXTENSION is
# idempotent (IF NOT EXISTS).

_CREATE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Inference log.  Each row is one prediction returned by the serving service.
-- uuid is provided by the client when the origin sample is known; otherwise
-- gen_random_uuid() assigns a fresh value so every prediction is recorded.
CREATE TABLE IF NOT EXISTS predictions (
    uuid                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp               TIMESTAMPTZ NOT NULL,
    model_version           TEXT        NOT NULL,
    prediction              INTEGER     NOT NULL,
    confidence              REAL        NOT NULL,
    prediction_distribution REAL[]      NOT NULL,
    embedding               REAL[]      NOT NULL,
    annotation_status       TEXT        NOT NULL DEFAULT 'none'
                            CHECK (annotation_status IN ('none', 'candidate', 'annotated')),
    annotated_label         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON predictions (timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version
    ON predictions (model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_annotation_status
    ON predictions (annotation_status);

-- Versioned dataset membership.  Each row assigns one sample UUID to a named
-- dataset version and split.  Label and MinIO path are stored here directly —
-- there is no separate samples oracle table.
CREATE TABLE IF NOT EXISTS dataset_samples (
    uuid        UUID        NOT NULL,
    version_id  TEXT        NOT NULL,
    split       TEXT        NOT NULL CHECK (split IN ('train', 'val', 'test')),
    label       INTEGER     NOT NULL,
    minio_path  TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (uuid, version_id)
);
CREATE INDEX IF NOT EXISTS idx_dataset_samples_version_split
    ON dataset_samples (version_id, split);
"""

# ── SQL — predictions table ───────────────────────────────────────────────────

_INSERT = """
INSERT INTO predictions (
    uuid, timestamp, model_version,
    prediction, confidence, prediction_distribution, embedding,
    annotation_status, annotated_label
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (uuid) DO NOTHING;
"""

_SELECT_WINDOW = """
SELECT uuid, timestamp, model_version,
       prediction, confidence, prediction_distribution, embedding,
       annotation_status, annotated_label
FROM predictions
WHERE timestamp >= %s
  AND (%s IS NULL OR timestamp < %s)
  AND (%s IS NULL OR model_version = %s)
ORDER BY timestamp;
"""

_SELECT_LABELED = """
SELECT uuid, timestamp, model_version,
       prediction, confidence, prediction_distribution, embedding,
       annotation_status, annotated_label
FROM predictions
WHERE annotated_label IS NOT NULL AND timestamp >= %s
ORDER BY timestamp;
"""

_MARK_CANDIDATE = """
UPDATE predictions SET annotation_status = 'candidate'
WHERE uuid = %s AND annotation_status = 'none';
"""

_MARK_CANDIDATES_BATCH = """
UPDATE predictions
SET annotation_status = 'candidate'
WHERE uuid IN (
    SELECT uuid
    FROM predictions
    WHERE annotation_status = 'none'
    ORDER BY RANDOM()
    LIMIT %s
)
RETURNING uuid;
"""

_WRITE_LABEL = """
UPDATE predictions
SET annotated_label = %s, annotation_status = 'annotated'
WHERE uuid = %s;
"""

_COUNT_LABELS = """
SELECT COUNT(*) FROM predictions
WHERE annotated_label IS NOT NULL AND timestamp >= %s;
"""

# Annotated predictions whose UUIDs are not yet part of any dataset version.
# Used by the drift-monitoring service to surface how many newly-annotated
# samples are available for the next training run.
_COUNT_ANNOTATED = """
SELECT COUNT(*)
FROM predictions
WHERE annotation_status = 'annotated'
  AND uuid NOT IN (SELECT uuid FROM dataset_samples);
"""

def _row_to_record(row: tuple) -> PredictRecord:
    (
        uuid, timestamp, model_version,
        prediction, confidence, prediction_distribution, embedding,
        annotation_status, annotated_label,
    ) = row
    return PredictRecord(
        uuid=uuid,
        timestamp=timestamp,
        model_version=model_version,
        # psycopg2 returns REAL[] as a list; list() guards against any adapter variance
        embedding=list(embedding),
        prediction=prediction,
        confidence=confidence,
        prediction_distribution=list(prediction_distribution),
        annotation_status=annotation_status,
        annotated_label=annotated_label,
    )


# ── Base ──────────────────────────────────────────────────────────────────────

class _DataControllerBase:
    """Postgres connection lifecycle and schema creation.

    All service-specific controllers inherit from this class.  psycopg2 and
    psycopg2.extras are imported lazily so services that don't use the data
    controller don't need them installed at import time.

    On first connection the UUID type adapter is registered so Python
    ``uuid.UUID`` objects round-trip correctly through Postgres UUID columns.
    """

    def __init__(self, dsn: str) -> None:
        import psycopg2  # lazy — keeps psycopg2 out of import-time for non-users
        import psycopg2.extras  # extras is a submodule; must be imported explicitly
        self._psycopg2 = psycopg2
        self._conn = None
        self._dsn = dsn
        self._ensure_schema()

    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg2.connect(self._dsn)
            # Register UUID ↔ uuid.UUID adaptation for this connection so that
            # Python UUID objects are sent as native Postgres UUID values.
            self._psycopg2.extras.register_uuid(self._conn)
        return self._conn

    def _ensure_schema(self) -> None:
        """Create both tables if they do not already exist."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_CREATE_SCHEMA)
            conn.commit()
        except Exception as exc:
            raise DataControllerError(
                f"Failed to create database schema: {exc}"
            ) from exc
