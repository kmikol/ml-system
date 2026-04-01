# shared/data_controller/serving.py
"""ServingDataController — fire-and-forget prediction persistence for the serving service."""

from __future__ import annotations

import logging
import os

import numpy as np

from shared.data_controller._base import _INSERT, _DataControllerBase
from shared.data_controller._object_store import MinIOObjectStore
from shared.schemas.predict_record import PredictRecord

logger = logging.getLogger(__name__)


class ServingDataController(_DataControllerBase):
    """Used by the serving service to persist prediction records.

    Graceful degradation: if Postgres or MinIO are unavailable at startup,
    logs a warning and silently skips the corresponding writes. Failures after
    connection are also swallowed — serving never raises due to storage issues.

    Instantiate at module level; no separate connect() call is needed.
    """

    def __init__(self) -> None:
        self._available = False
        self._failures = 0
        self._s3_available = False
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

        # Object store for prediction image storage — optional, same fire-and-forget semantics.
        # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY come from the ml-system-secrets K8s Secret.
        endpoint = os.getenv("DATASET_S3_ENDPOINT_URL", "")
        bucket = os.getenv("DATASET_BUCKET", "")
        if not endpoint or not bucket:
            logger.warning(
                "DATASET_S3_ENDPOINT_URL or DATASET_BUCKET not set, "
                "prediction images will not be stored in the object store"
            )
        else:
            try:
                self._store = MinIOObjectStore(
                    endpoint_url=endpoint,
                    bucket=bucket,
                    access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                    secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                )
                self._s3_available = True
                logger.info("Object store ready for prediction image storage")
            except Exception as exc:
                logger.warning(f"Object store setup failed (serving continues without): {exc}")

    def store_prediction(self, record: PredictRecord, image_2d: list | None = None) -> None:
        """Persist a prediction record to Postgres and, if provided, its image to object storage.

        Fire-and-forget: if either store fails, the error is logged as a warning
        and swallowed. Serving never raises due to storage issues.

        Args:
            record:   Fully populated PredictRecord.
            image_2d: Optional 14×14 nested list of float32 pixel values [0, 1].
                      Stored at predictions/{record.uuid}.npy.
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

        if image_2d is not None and self._s3_available:
            try:
                self._store.put_array(
                    f"predictions/{record.uuid}.npy",
                    np.array(image_2d, dtype=np.float32),
                )
            except Exception as exc:
                logger.warning(f"Prediction image upload failed for {record.uuid}: {exc}")
