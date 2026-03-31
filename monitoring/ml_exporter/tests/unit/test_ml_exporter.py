# monitoring/ml_exporter/tests/unit/test_ml_exporter.py
"""
Unit tests for monitoring/ml_exporter/main.py.

No environment variables, no DB, no MLflow, no Prometheus global state required.
All external dependencies are replaced with in-process fakes.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from monitoring.ml_exporter.main import (
    DriftPoller,
    ExporterConfig,
    WindowMetrics,
    compute_psi,
    compute_window_metrics,
)
from shared.model_artifact_controller import ModelArtifactError
from shared.schemas.predict_record import PredictRecord

# ── Fakes ──────────────────────────────────────────────────────────────────────


def _make_config(**overrides: Any) -> ExporterConfig:
    defaults = {
        "model_name": "test-model",
        "model_stage": "Production",
        "poll_interval": 60,
        "window_seconds": 3600,
    }
    defaults.update(overrides)
    return ExporterConfig(**defaults)


def _make_record(prediction: int = 0, confidence: float = 0.9) -> PredictRecord:
    return PredictRecord(
        uuid=uuid4(),
        timestamp=datetime.now(UTC),
        model_version="v1",
        embedding=[0.0] * 64,
        prediction=prediction,
        confidence=confidence,
        prediction_distribution=[0.1] * 10,
    )


class FakeArtifactController:
    def __init__(
        self,
        run_id: str = "run-1",
        reference: dict[str, Any] | None = None,
        fail_get_run_id: bool = False,
        fail_download: bool = False,
    ) -> None:
        self.run_id = run_id
        self.reference = reference or {"prediction_class_frequencies": [0.1] * 10}
        self.fail_get_run_id = fail_get_run_id
        self.fail_download = fail_download

    def get_production_run_id(self, model_name: str, stage: str) -> str:
        if self.fail_get_run_id:
            raise ModelArtifactError("MLflow unavailable")
        return self.run_id

    def download_reference_distribution(self, run_id: str, local_dir: str) -> dict[str, Any]:
        if self.fail_download:
            raise Exception("Download failed")
        return self.reference


class FakeDriftDataController:
    def __init__(
        self, records: list[PredictRecord] | None = None, annotated_count: int = 0
    ) -> None:
        self.records = records or []
        self.annotated_count = annotated_count
        self.fail_get_predictions = False

    def get_predictions(
        self, since: datetime, until: datetime | None = None, model_version: str | None = None
    ) -> list[PredictRecord]:
        if self.fail_get_predictions:
            raise Exception("DB unavailable")
        return self.records

    def get_annotated_count(self) -> int:
        return self.annotated_count


class FakeEmitter:
    def __init__(self) -> None:
        self.emitted: list[tuple[WindowMetrics, int]] = []
        self.age_updates: list[float] = []

    def emit(self, metrics: WindowMetrics, annotated_count: int) -> None:
        self.emitted.append((metrics, annotated_count))

    def update_poll_age(self, age: float) -> None:
        self.age_updates.append(age)

    def generate_metrics(self) -> tuple[bytes, str]:
        return b"", "text/plain"


def _make_poller(
    config: ExporterConfig | None = None,
    data: FakeDriftDataController | None = None,
    artifacts: FakeArtifactController | None = None,
    emitter: FakeEmitter | None = None,
) -> tuple[DriftPoller, FakeEmitter]:
    em = emitter or FakeEmitter()
    poller = DriftPoller(
        config=config or _make_config(),
        data_controller=data or FakeDriftDataController(),
        artifact_controller=artifacts or FakeArtifactController(),
        emitter=em,
        artifact_dir="/tmp",
    )
    return poller, em


# ── compute_psi ────────────────────────────────────────────────────────────────


class TestComputePsi:
    def test_identical_distributions_return_zero(self):
        dist = [0.1] * 10
        assert compute_psi(dist, dist) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_is_positive(self):
        actual = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
        reference = [0.1] * 10
        assert compute_psi(actual, reference) > 0.0

    def test_large_shift_exceeds_threshold(self):
        # Concentrate all mass on class 0 vs uniform — PSI should be > 0.25.
        actual = [1.0] + [0.0] * 9
        reference = [0.1] * 10
        assert compute_psi(actual, reference) > 0.25

    def test_zero_probability_class_handled_via_epsilon(self):
        # Both actual and reference have a zero-probability class; must not raise.
        actual = [0.2, 0.8] + [0.0] * 8
        reference = [0.1] * 10
        result = compute_psi(actual, reference)
        assert math.isfinite(result)

    def test_both_zero_class_contributes_nothing(self):
        # When both actual[c] and reference[c] are 0, the term ≈ 0.
        actual = [0.5, 0.5] + [0.0] * 8
        reference = [0.5, 0.5] + [0.0] * 8
        assert compute_psi(actual, reference) == pytest.approx(0.0, abs=1e-6)


# ── compute_window_metrics ─────────────────────────────────────────────────────


class TestComputeWindowMetrics:
    def test_empty_records_returns_zero_counts(self):
        m = compute_window_metrics([], reference=None)
        assert m.n == 0
        assert all(m.class_counts[c] == 0 for c in range(10))
        assert all(m.class_freqs[c] == 0.0 for c in range(10))
        assert m.confidence_mean == 0.0
        assert m.psi is None

    def test_psi_is_none_without_reference(self):
        records = [_make_record(prediction=3) for _ in range(50)]
        m = compute_window_metrics(records, reference=None)
        assert m.psi is None

    def test_psi_is_none_below_min_samples(self):
        records = [_make_record() for _ in range(5)]  # below _MIN_SAMPLES = 30
        reference = [0.1] * 10
        m = compute_window_metrics(records, reference=reference)
        assert m.psi is None

    def test_psi_computed_above_min_samples(self):
        reference = [0.1] * 10
        records = [_make_record(prediction=0) for _ in range(30)]  # exactly at threshold
        m = compute_window_metrics(records, reference=reference)
        assert m.psi is not None
        assert math.isfinite(m.psi)

    def test_class_counts_and_freqs_match_records(self):
        records = [_make_record(prediction=0)] * 3 + [_make_record(prediction=1)] * 7
        m = compute_window_metrics(records, reference=None)
        assert m.n == 10
        assert m.class_counts[0] == 3
        assert m.class_counts[1] == 7
        assert m.class_freqs[0] == pytest.approx(0.3)
        assert m.class_freqs[1] == pytest.approx(0.7)

    def test_confidence_mean_is_average_of_records(self):
        records = [_make_record(confidence=0.8), _make_record(confidence=0.6)]
        m = compute_window_metrics(records, reference=None)
        assert m.confidence_mean == pytest.approx(0.7)

    def test_identical_actual_and_reference_gives_zero_psi(self):
        reference = [0.1] * 10
        # 3 records per class, 10 classes = 30 records, uniform distribution.
        records = [_make_record(prediction=c) for c in range(10) for _ in range(3)]
        m = compute_window_metrics(records, reference=reference)
        assert m.psi is not None
        assert m.psi == pytest.approx(0.0, abs=1e-6)


# ── DriftPoller ────────────────────────────────────────────────────────────────


class TestDriftPollerObservers:
    def test_current_run_id_is_none_before_poll(self):
        poller, _ = _make_poller(artifacts=FakeArtifactController(fail_get_run_id=True))
        assert poller.current_run_id() is None

    def test_reference_loaded_is_false_before_poll(self):
        poller, _ = _make_poller(artifacts=FakeArtifactController(fail_get_run_id=True))
        assert poller.reference_loaded() is False

    def test_poll_age_is_none_before_first_poll(self):
        poller, _ = _make_poller()
        assert poller.poll_age() is None

    def test_poll_age_is_float_after_successful_poll(self):
        poller, _ = _make_poller()
        poller.poll()
        age = poller.poll_age()
        assert age is not None
        assert age >= 0.0

    def test_current_run_id_set_after_poll(self):
        poller, _ = _make_poller(artifacts=FakeArtifactController(run_id="my-run"))
        poller.poll()
        assert poller.current_run_id() == "my-run"

    def test_reference_loaded_after_successful_poll(self):
        poller, _ = _make_poller()
        poller.poll()
        assert poller.reference_loaded() is True


class TestDriftPollerPoll:
    def test_emits_metrics_on_success(self):
        records = [_make_record(prediction=c) for c in range(10) for _ in range(3)]
        data = FakeDriftDataController(records=records, annotated_count=5)
        poller, em = _make_poller(data=data)
        poller.poll()
        assert len(em.emitted) == 1
        metrics, annotated = em.emitted[0]
        assert metrics.n == 30
        assert annotated == 5

    def test_does_not_emit_on_db_failure(self):
        data = FakeDriftDataController()
        data.fail_get_predictions = True
        poller, em = _make_poller(data=data)
        poller.poll()
        assert len(em.emitted) == 0

    def test_poll_age_not_updated_on_db_failure(self):
        data = FakeDriftDataController()
        data.fail_get_predictions = True
        poller, _ = _make_poller(data=data)
        poller.poll()
        assert poller.poll_age() is None

    def test_emits_with_zero_annotated_when_count_fails(self):
        class BrokenCountController(FakeDriftDataController):
            def get_annotated_count(self) -> int:
                raise Exception("count unavailable")

        data = BrokenCountController(records=[_make_record()])
        poller, em = _make_poller(data=data)
        poller.poll()
        assert len(em.emitted) == 1
        _, annotated = em.emitted[0]
        assert annotated == 0

    def test_emits_with_psi_none_when_reference_unavailable(self):
        data = FakeDriftDataController(records=[_make_record()] * 30)
        artifacts = FakeArtifactController(fail_get_run_id=True)
        poller, em = _make_poller(data=data, artifacts=artifacts)
        poller.poll()
        metrics, _ = em.emitted[0]
        assert metrics.psi is None

    def test_emits_with_psi_when_reference_loaded(self):
        reference = [0.1] * 10
        records = [_make_record(prediction=c) for c in range(10) for _ in range(3)]
        artifacts = FakeArtifactController(reference={"prediction_class_frequencies": reference})
        data = FakeDriftDataController(records=records)
        poller, em = _make_poller(data=data, artifacts=artifacts)
        poller.poll()
        metrics, _ = em.emitted[0]
        assert metrics.psi is not None
        assert math.isfinite(metrics.psi)

    def test_reloads_reference_when_run_id_changes(self):
        artifacts = FakeArtifactController(run_id="run-1")
        poller, em = _make_poller(artifacts=artifacts)

        poller.poll()
        assert poller.current_run_id() == "run-1"

        artifacts.run_id = "run-2"
        poller.poll()
        assert poller.current_run_id() == "run-2"
        assert len(em.emitted) == 2

    def test_uses_cached_reference_when_mlflow_unavailable(self):
        reference = [0.1] * 10
        artifacts = FakeArtifactController(reference={"prediction_class_frequencies": reference})
        records = [_make_record(prediction=c) for c in range(10) for _ in range(3)]
        data = FakeDriftDataController(records=records)
        poller, em = _make_poller(data=data, artifacts=artifacts)

        # First poll loads reference successfully.
        poller.poll()
        assert poller.reference_loaded() is True

        # MLflow goes down — reference should persist.
        artifacts.fail_get_run_id = True
        poller.poll()
        metrics, _ = em.emitted[1]
        assert metrics.psi is not None  # cached reference still used

    def test_reference_download_failure_clears_reference(self):
        artifacts = FakeArtifactController(fail_download=True)
        poller, _ = _make_poller(artifacts=artifacts)
        poller.poll()
        # Download failed → reference not loaded, but no exception raised.
        assert poller.reference_loaded() is False

    def test_empty_window_emits_zero_counts(self):
        poller, em = _make_poller(data=FakeDriftDataController(records=[]))
        poller.poll()
        metrics, _ = em.emitted[0]
        assert metrics.n == 0
        assert all(metrics.class_counts[c] == 0 for c in range(10))
