# shared/data_controller/sampling.py
"""SamplingDataController — query and annotate predictions for sample selection."""

from __future__ import annotations

from datetime import datetime

from shared.config import require_env
from shared.data_controller._base import (
    _COUNT_LABELS,
    _MARK_CANDIDATE,
    _SELECT_WINDOW,
    DataControllerError,
    _DataControllerBase,
    _row_to_record,
)
from shared.schemas.predict_record import PredictRecord


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
