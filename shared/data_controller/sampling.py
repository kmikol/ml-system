# shared/data_controller/sampling.py
"""SamplingDataController — atomic candidate selection for the annotation pipeline."""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from shared.config import require_env
from shared.data_controller._base import (
    _MARK_CANDIDATES_BATCH,
    _MARK_CANDIDATES_LOW_CONFIDENCE,
    _MARK_CANDIDATES_HIGH_MAHALANOBIS,
    _MARK_CANDIDATES_DIVERSE,
    DataControllerError,
    _DataControllerBase,
)

logger = logging.getLogger(__name__)

SamplingStrategy = Literal["random", "low_confidence", "high_mahalanobis", "diverse"]


class SamplingDataController(_DataControllerBase):
    """Used by the sampling job to atomically select and mark annotation candidates.

    Any prediction with ``annotation_status='none'`` is eligible.  Selection strategy
    is configurable via the strategy parameter. Supports:
    - random: Random selection (default, backward compatible)
    - low_confidence: Prioritizes predictions with low confidence scores
    - high_mahalanobis: Prioritizes predictions with high Mahalanobis distance
    - diverse: Combined strategy that balances confidence and Mahalanobis distance
               while ensuring diversity through bucketing

    The annotation job resolves labels for each candidate UUID from the file-based oracle.
    """

    def __init__(self) -> None:
        super().__init__(require_env("DATA_CONTROLLER_DB_URL"))

    def select_and_mark_candidates(
        self, limit: int, strategy: SamplingStrategy = "random"
    ) -> list[UUID]:
        """Atomically select up to *limit* unannotated predictions and advance
        their ``annotation_status`` to ``'candidate'``.

        Args:
            limit: Maximum number of candidates to select
            strategy: Sampling strategy to use (random, low_confidence,
                     high_mahalanobis, or diverse)

        Returns:
            List of prediction UUIDs that were marked as candidate.
        """
        # Select the appropriate query based on strategy
        if strategy == "random":
            query = _MARK_CANDIDATES_BATCH
            params = (limit,)
        elif strategy == "low_confidence":
            query = _MARK_CANDIDATES_LOW_CONFIDENCE
            params = (limit,)
        elif strategy == "high_mahalanobis":
            query = _MARK_CANDIDATES_HIGH_MAHALANOBIS
            params = (limit,)
        elif strategy == "diverse":
            query = _MARK_CANDIDATES_DIVERSE
            params = (limit, limit)
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to 'random'")
            query = _MARK_CANDIDATES_BATCH
            params = (limit,)

        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(query, params)
                marked = [row[0] for row in cur.fetchall()]
            conn.commit()
            logger.info(f"Strategy '{strategy}' selected {len(marked)} candidates")
            return marked
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            raise DataControllerError(f"Failed to mark candidates: {exc}") from exc
