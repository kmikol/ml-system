# shared/data_controller/annotation.py
"""AnnotationDataController — write ground truth labels."""

from __future__ import annotations

from uuid import UUID

from shared.config import require_env
from shared.data_controller._base import (
    _RESET_CANDIDATE,
    _WRITE_LABEL,
    DataControllerError,
    _DataControllerBase,
)

# Return the UUIDs of candidate predictions, randomly ordered so each job run
# annotates a different subset.  Labels are resolved by the annotation job from
# the file-based oracle (uuids.npy + labels.npy), not from the database.
_GET_CANDIDATES = """
SELECT uuid FROM predictions
WHERE annotation_status = 'candidate'
ORDER BY RANDOM()
LIMIT %s;
"""


class AnnotationDataController(_DataControllerBase):
    """Used by the annotation service to write ground truth labels."""

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))

    def get_candidates(self, limit: int) -> list[UUID]:
        """Return up to *limit* candidate prediction UUIDs.

        Results are randomly ordered so each job run annotates a different
        subset.  The caller is responsible for resolving labels via the
        file-based oracle.

        Args:
            limit: Maximum number of candidates to return.

        Returns:
            List of prediction UUIDs with ``annotation_status='candidate'``.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_GET_CANDIDATES, (limit,))
                return [row[0] for row in cur.fetchall()]
        except Exception as exc:
            raise DataControllerError(f"Failed to fetch candidates: {exc}") from exc

    def write_label(self, uuid: UUID, label: int) -> None:
        """Write a ground truth label and advance annotation_status to 'annotated'.

        Only updates predictions that are currently in 'candidate' status.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_WRITE_LABEL, (label, uuid))
            conn.commit()
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(f"Failed to write label for '{uuid}': {exc}") from exc

    def reset_candidate(self, uuid: UUID) -> None:
        """Reset a candidate prediction back to 'none' so it can be re-sampled.

        Used when the annotation oracle has no label for this UUID.
        Only affects predictions currently in 'candidate' status.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_RESET_CANDIDATE, (uuid,))
            conn.commit()
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(f"Failed to reset candidate '{uuid}': {exc}") from exc
