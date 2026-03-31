# shared/data_controller/sampling.py
"""SamplingDataController — atomic candidate selection for the annotation pipeline."""

from __future__ import annotations

from uuid import UUID

from shared.config import require_env
from shared.data_controller._base import (
    _MARK_CANDIDATES_BATCH,
    DataControllerError,
    _DataControllerBase,
)


class SamplingDataController(_DataControllerBase):
    """Used by the sampling job to atomically select and mark annotation candidates.

    Any prediction with ``annotation_status='none'`` is eligible.  Selection is
    random so repeated runs sample different subsets.  The annotation job resolves
    labels for each candidate UUID from the file-based oracle.
    """

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))

    def select_and_mark_candidates(self, limit: int) -> list[UUID]:
        """Atomically select up to *limit* unannotated predictions and advance
        their ``annotation_status`` to ``'candidate'``.

        Selection is random so repeated runs sample different subsets.

        Returns:
            List of prediction UUIDs that were marked as candidate.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_MARK_CANDIDATES_BATCH, (limit,))
                marked = [row[0] for row in cur.fetchall()]
            conn.commit()
            return marked
        except Exception as exc:
            with self._conn_lock:
                try:
                    self._conn.rollback()
                except Exception:
                    self._conn = None
            raise DataControllerError(f"Failed to mark candidates: {exc}") from exc
