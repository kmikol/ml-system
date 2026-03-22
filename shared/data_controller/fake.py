# shared/data_controller/fake.py
"""FakeDataController — in-memory implementation for unit tests."""

from __future__ import annotations

from datetime import datetime

from shared.schemas.predict_record import PredictRecord


class FakeDataController:
    """In-memory implementation for unit tests — no Postgres required.

    Implements the full surface area of all service controllers. Use in place
    of any service-specific controller by duck typing.
    """

    def __init__(self) -> None:
        self._records: list[PredictRecord] = []

    def store_prediction(self, record: PredictRecord) -> None:
        self._records.append(record.model_copy())

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
        return [r for r in self._records if r.label is not None and r.timestamp >= since]

    def mark_candidate(self, prediction_id: str) -> None:
        for r in self._records:
            if r.prediction_id == prediction_id and r.annotation_status == "none":
                r.annotation_status = "candidate"
                return

    def write_label(self, prediction_id: str, label: int) -> None:
        for r in self._records:
            if r.prediction_id == prediction_id:
                r.label = label
                r.annotation_status = "annotated"
                return

    def count_labels_since(self, since: datetime) -> int:
        return sum(
            1 for r in self._records
            if r.label is not None and r.timestamp >= since
        )
