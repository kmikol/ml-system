# shared/data_controller/serving.py
"""ServingDataController — fire-and-forget prediction persistence for the serving service."""

from __future__ import annotations

import logging
import os

from shared.data_controller._base import _INSERT, _DataControllerBase
from shared.schemas.predict_record import PredictRecord

logger = logging.getLogger(__name__)


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
            logger.warning("DATA_CONTROLLER_DB_URL not set, predictions will not be persisted")
            return
        try:
            super().__init__(dsn)
            self._available = True
            logger.info("Data controller connected")
        except Exception as exc:
            logger.warning(f"Data controller unavailable (serving continues without): {exc}")

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
                cur.execute(
                    _INSERT,
                    (
                        record.uuid,
                        record.timestamp,
                        record.model_version,
                        record.prediction,
                        record.confidence,
                        record.prediction_distribution,  # list[float] → psycopg2 → REAL[]
                        record.embedding,  # list[float] → psycopg2 → REAL[]
                        record.mahalanobis_distance,
                        record.annotation_status,
                        record.annotated_label,
                    ),
                )
            conn.commit()
        except Exception as exc:
            self._failures += 1
            logger.warning(f"Prediction storage failed ({self._failures} total): {exc}")
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
