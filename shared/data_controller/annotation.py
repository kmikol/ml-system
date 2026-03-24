# shared/data_controller/annotation.py
"""AnnotationDataController — write ground truth labels."""

from __future__ import annotations

from shared.config import require_env
from shared.data_controller._base import _WRITE_LABEL, DataControllerError, _DataControllerBase

# Randomly sample up to *limit* candidate predictions that have a matching entry
# in dataset_samples (joined on the UUID that propagates from dataset → serving →
# predictions table).  ORDER BY RANDOM() gives a uniform sample each run.
_GET_CANDIDATES = """
SELECT p.prediction_id, d.label
FROM predictions p
JOIN dataset_samples d ON d.sample_id = p.prediction_id
WHERE p.annotation_status = 'candidate'
ORDER BY RANDOM()
LIMIT %s;
"""


class AnnotationDataController(_DataControllerBase):
    """Used by the annotation service to write ground truth labels."""

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))

    def get_candidates(self, limit: int) -> list[tuple[str, int]]:
        """Return up to *limit* candidate predictions with their ground truth labels.

        Joins the predictions table (annotation_status='candidate') with
        dataset_samples on the UUID to retrieve the ground truth label for each
        candidate.  Results are randomly ordered so each job run annotates a
        different subset.

        Args:
            limit: Maximum number of candidates to return.

        Returns:
            List of ``(prediction_id, ground_truth_label)`` tuples.
        """
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_GET_CANDIDATES, (limit,))
                return [(row[0], row[1]) for row in cur.fetchall()]
        except Exception as exc:
            raise DataControllerError(f"Failed to fetch candidates: {exc}") from exc

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
