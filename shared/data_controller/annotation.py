# shared/data_controller/annotation.py
"""AnnotationDataController — write ground truth labels."""

from __future__ import annotations

from shared.config import require_env
from shared.data_controller._base import _WRITE_LABEL, DataControllerError, _DataControllerBase


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
