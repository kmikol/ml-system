# shared/data_controller/fake.py
"""FakeDataController — in-memory implementation for unit tests."""

from __future__ import annotations

import random
from datetime import datetime
from uuid import UUID

from shared.schemas.predict_record import PredictRecord


class FakeDataController:
    """In-memory implementation for unit tests — no Postgres required.

    Implements the full surface area of all service controllers. Use in place
    of any service-specific controller by duck typing.
    """

    def __init__(self) -> None:
        self._records: list[PredictRecord] = []

    # ── ServingDataController surface ─────────────────────────────────────────

    def store_prediction(self, record: PredictRecord) -> None:
        self._records.append(record.model_copy())

    # ── DriftDataController surface ───────────────────────────────────────────

    def get_predictions(
        self,
        since: datetime,
        until: datetime | None = None,
        model_version: str | None = None,
    ) -> list[PredictRecord]:
        result = [r for r in self._records if r.timestamp >= since]
        if until is not None:
            result = [r for r in result if r.timestamp < until]
        if model_version is not None:
            result = [r for r in result if r.model_version == model_version]
        return result

    def get_labeled_predictions(self, since: datetime) -> list[PredictRecord]:
        return [r for r in self._records if r.annotated_label is not None and r.timestamp >= since]

    def get_annotated_count(self) -> int:
        """Return annotated predictions count (simulates _COUNT_ANNOTATED query).

        In-memory approximation: counts all annotated records, since the fake
        has no concept of dataset_samples membership.
        """
        return sum(1 for r in self._records if r.annotation_status == "annotated")

    # ── SamplingDataController surface ────────────────────────────────────────

    def mark_candidate(self, uuid: UUID) -> None:
        for r in self._records:
            if r.uuid == uuid and r.annotation_status == "none":
                r.annotation_status = "candidate"
                return

    def select_and_mark_candidates(self, limit: int) -> list[UUID]:
        """Atomically select and mark eligible predictions as candidates.

        Any prediction with ``annotation_status='none'`` is eligible.
        """
        eligible = [r for r in self._records if r.annotation_status == "none"]
        random.shuffle(eligible)
        selected = eligible[:limit]
        for r in selected:
            r.annotation_status = "candidate"
        return [r.uuid for r in selected]

    # ── AnnotationDataController surface ──────────────────────────────────────

    def get_candidates(self, limit: int) -> list[UUID]:
        """Return up to *limit* candidate prediction UUIDs.

        Labels are resolved by the caller from the file-based oracle, not
        from the in-memory store.
        """
        candidates = [r.uuid for r in self._records if r.annotation_status == "candidate"]
        random.shuffle(candidates)
        return candidates[:limit]

    def write_label(self, uuid: UUID, label: int) -> None:
        for r in self._records:
            if r.uuid == uuid and r.annotation_status == "candidate":
                r.annotated_label = label
                r.annotation_status = "annotated"
                return

    def reset_candidate(self, uuid: UUID) -> None:
        for r in self._records:
            if r.uuid == uuid and r.annotation_status == "candidate":
                r.annotation_status = "none"
                return
