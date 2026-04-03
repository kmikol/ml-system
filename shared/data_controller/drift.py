# shared/data_controller/drift.py
"""DriftDataController — query prediction history for drift monitoring."""

from __future__ import annotations

from datetime import datetime

from shared.config import require_env
from shared.data_controller._base import (
    _COUNT_ANNOTATED,
    _SELECT_LABELED,
    _SELECT_WINDOW,
    DataControllerError,
    _DataControllerBase,
    _row_to_record,
)
from shared.schemas.predict_record import PredictRecord


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
                records = [_row_to_record(row) for row in cur.fetchall()]
            conn.commit()
            return records
        except Exception as exc:
            raise DataControllerError(f"Failed to query predictions: {exc}") from exc

    def get_annotated_count(self) -> int:
        """Return annotated predictions whose UUIDs are not yet in any dataset version.

        This count surfaces how many newly-annotated samples are available to
        include in the next training run.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_COUNT_ANNOTATED)
                count = cur.fetchone()[0]
            conn.commit()
            return count
        except Exception as exc:
            raise DataControllerError(f"Failed to count annotated predictions: {exc}") from exc

    def get_labeled_predictions(self, since: datetime) -> list[PredictRecord]:
        """Return predictions that have a ground truth label, since *since*."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_SELECT_LABELED, (since,))
                records = [_row_to_record(row) for row in cur.fetchall()]
            conn.commit()
            return records
        except Exception as exc:
            raise DataControllerError(f"Failed to query labeled predictions: {exc}") from exc
