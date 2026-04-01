# tests/unit/test_ml_exporter.py
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
    VersionPsiResult,
    WindowMetrics,
    compute_psi,
    compute_window_metrics,
)
from shared.model_artifact_controller import ModelArtifactError, ModelStage, ReferenceDistribution
from shared.model_artifact_controller.types import VersionArtifacts
from shared.schemas.predict_record import PredictRecord

# ── Fakes ──────────────────────────────────────────────────────────────────────


def _make_config(**overrides: Any) -> ExporterConfig:
    defaults = {
        "model_stage": ModelStage.PRODUCTION,
        "poll_interval": 60,
        "window_seconds": 3600,
        "reference_cache_ttl_seconds": 300,
    }
    defaults.update(overrides)
    return ExporterConfig(**defaults)


def _make_record(
    prediction: int = 0,
    confidence: float = 0.9,
    model_version: str = "v1",
) -> PredictRecord:
    return PredictRecord(
        uuid=uuid4(),
        timestamp=datetime.now(UTC),
        model_version=model_version,
        embedding=[0.0] * 64,
        prediction=prediction,
        confidence=confidence,
        prediction_distribution=[0.1] * 10,
    )


def _make_reference(freqs: list[float] | None = None) -> ReferenceDistribution:
    return ReferenceDistribution(
        num_samples=100,
        pixel_mean=0.5,
        pixel_std=0.2,
        embedding_mean=[0.0] * 64,
        embedding_cov=[[0.0] * 64] * 64,
        prediction_class_frequencies=freqs or [0.1] * 10,
    )


class FakeModelStore:
    """Fake ModelStore for unit testing the DriftPoller."""

    def __init__(
        self,
        version_id: str = "run-1",
        reference: ReferenceDistribution | None = None,
        fail_get_version_id: bool = False,
        fail_download: bool = False,
    ) -> None:
        self.version_id = version_id
        self.reference = reference or _make_reference()
        self.fail_get_version_id = fail_get_version_id
        self.fail_download = fail_download
        # Track how many times each version was fetched.
        self.fetch_counts: dict[str, int] = {}

    def get_current_version_id(self, stage: ModelStage = ModelStage.PRODUCTION) -> str:
        if self.fail_get_version_id:
            raise ModelArtifactError("Registry unavailable")
        return self.version_id

    def get_reference_distribution(
        self, stage: ModelStage = ModelStage.PRODUCTION
    ) -> ReferenceDistribution:
        if self.fail_download:
            raise Exception("Download failed")
        return self.reference

    def get_version_artifacts(
        self,
        version_id: str,
        *,
        include_gaussians: bool = False,
        include_reference: bool = False,
    ) -> VersionArtifacts:
        """Return a VersionArtifacts with optional reference distribution."""
        self.fetch_counts[version_id] = self.fetch_counts.get(version_id, 0) + 1
        if self.fail_download:
            raise Exception("Download failed")
        ref = self.reference if include_reference else None
        return VersionArtifacts(
            model_path="/fake/model.onnx",
            class_gaussians=None,
            reference_distribution=ref,
        )


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
        self.emitted: list[tuple[WindowMetrics, int, list[VersionPsiResult]]] = []
        self.age_updates: list[float] = []

    def emit(
        self,
        metrics: WindowMetrics,
        annotated_count: int,
        version_psi_results: list[VersionPsiResult],
    ) -> None:
        self.emitted.append((metrics, annotated_count, version_psi_results))

    def update_poll_age(self, age: float) -> None:
        self.age_updates.append(age)

    def generate_metrics(self) -> tuple[bytes, str]:
        return b"", "text/plain"


def _make_poller(
    config: ExporterConfig | None = None,
    data: FakeDriftDataController | None = None,
    store: FakeModelStore | None = None,
    emitter: FakeEmitter | None = None,
) -> tuple[DriftPoller, FakeEmitter]:
    em = emitter or FakeEmitter()
    poller = DriftPoller(
        config=config or _make_config(),
        data_controller=data or FakeDriftDataController(),
        store=store or FakeModelStore(),
        emitter=em,
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
        poller, _ = _make_poller(store=FakeModelStore(fail_get_version_id=True))
        assert poller.current_version_id() is None

    def test_reference_loaded_is_false_before_poll(self):
        poller, _ = _make_poller(store=FakeModelStore(fail_get_version_id=True))
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
        poller, _ = _make_poller(store=FakeModelStore(version_id="my-run"))
        poller.poll()
        assert poller.current_version_id() == "my-run"

    def test_reference_loaded_after_successful_poll(self):
        records = [_make_record(model_version="v1")]
        data = FakeDriftDataController(records=records)
        poller, _ = _make_poller(data=data)
        poller.poll()
        assert poller.reference_loaded() is True


class TestDriftPollerPoll:
    def test_emits_metrics_on_success(self):
        records = [_make_record(prediction=c) for c in range(10) for _ in range(3)]
        data = FakeDriftDataController(records=records, annotated_count=5)
        poller, em = _make_poller(data=data)
        poller.poll()
        assert len(em.emitted) == 1
        metrics, annotated, version_psi_results = em.emitted[0]
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
        _, annotated, _ = em.emitted[0]
        assert annotated == 0

    def test_emits_with_psi_none_when_reference_unavailable(self):
        records = [_make_record(model_version="v1") for _ in range(30)]
        data = FakeDriftDataController(records=records)
        store = FakeModelStore(fail_download=True)
        poller, em = _make_poller(data=data, store=store)
        poller.poll()
        _, _, version_psi_results = em.emitted[0]
        assert len(version_psi_results) == 1
        assert version_psi_results[0].psi is None

    def test_emits_with_psi_when_reference_loaded(self):
        records = [_make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)]
        store = FakeModelStore(reference=_make_reference([0.1] * 10))
        data = FakeDriftDataController(records=records)
        poller, em = _make_poller(data=data, store=store)
        poller.poll()
        _, _, version_psi_results = em.emitted[0]
        assert len(version_psi_results) == 1
        assert version_psi_results[0].psi is not None
        assert math.isfinite(version_psi_results[0].psi)

    def test_reloads_reference_when_run_id_changes(self):
        artifacts = FakeModelStore(version_id="run-1")
        poller, em = _make_poller(store=artifacts)

        poller.poll()
        assert poller.current_version_id() == "run-1"

        artifacts.version_id = "run-2"
        poller.poll()
        assert poller.current_version_id() == "run-2"
        assert len(em.emitted) == 2

    def test_uses_cached_reference_when_mlflow_unavailable(self):
        records = [_make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)]
        store = FakeModelStore(reference=_make_reference([0.1] * 10))
        data = FakeDriftDataController(records=records)
        poller, em = _make_poller(data=data, store=store)

        # First poll loads reference successfully.
        poller.poll()
        assert poller.reference_loaded() is True

        # Store goes down — stale cached reference should still be used.
        store.fail_download = True
        poller.poll()
        _, _, version_psi_results = em.emitted[1]
        assert version_psi_results[0].psi is not None  # stale cached reference still used

    def test_reference_download_failure_clears_reference(self):
        records = [_make_record(model_version="v1") for _ in range(30)]
        data = FakeDriftDataController(records=records)
        store = FakeModelStore(fail_download=True)
        poller, _ = _make_poller(data=data, store=store)
        poller.poll()
        # Download failed and no stale cache → no reference loaded.
        assert poller.reference_loaded() is False

    def test_empty_window_emits_zero_counts(self):
        poller, em = _make_poller(data=FakeDriftDataController(records=[]))
        poller.poll()
        metrics, _, version_psi_results = em.emitted[0]
        assert metrics.n == 0
        assert all(metrics.class_counts[c] == 0 for c in range(10))
        assert version_psi_results == []


# ── Multi-version window ───────────────────────────────────────────────────────


class TestMultiVersionWindow:
    def test_emits_psi_for_each_version(self):
        """Window with two versions → two VersionPsiResult entries."""
        records_v1 = [
            _make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)
        ]
        records_v2 = [
            _make_record(prediction=c, model_version="v2") for c in range(10) for _ in range(3)
        ]
        store = FakeModelStore()
        data = FakeDriftDataController(records=records_v1 + records_v2)
        poller, em = _make_poller(data=data, store=store)
        poller.poll()

        _, _, version_psi_results = em.emitted[0]
        versions_seen = {vr.version for vr in version_psi_results}
        assert versions_seen == {"v1", "v2"}

    def test_psi_not_mixed_across_versions(self):
        """Per-version PSI uses only that version's records, not the combined window."""
        # v1: uniform → PSI ≈ 0
        records_v1 = [
            _make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)
        ]
        # v2: all class 0 → high PSI
        records_v2 = [
            _make_record(prediction=0, model_version="v2") for _ in range(30)
        ]
        store = FakeModelStore(reference=_make_reference([0.1] * 10))
        data = FakeDriftDataController(records=records_v1 + records_v2)
        poller, em = _make_poller(data=data, store=store)
        poller.poll()

        _, _, version_psi_results = em.emitted[0]
        by_version = {vr.version: vr for vr in version_psi_results}

        assert by_version["v1"].psi is not None
        assert by_version["v2"].psi is not None
        # v1 uniform vs uniform reference → low PSI; v2 all-class-0 → high PSI
        assert by_version["v1"].psi == pytest.approx(0.0, abs=1e-6)
        assert by_version["v2"].psi > 0.25

    def test_version_sample_counts_are_correct(self):
        """n field of each VersionPsiResult reflects only that version's records."""
        records_v1 = [_make_record(model_version="v1") for _ in range(10)]
        records_v2 = [_make_record(model_version="v2") for _ in range(20)]
        data = FakeDriftDataController(records=records_v1 + records_v2)
        poller, em = _make_poller(data=data)
        poller.poll()

        _, _, version_psi_results = em.emitted[0]
        by_version = {vr.version: vr for vr in version_psi_results}
        assert by_version["v1"].n == 10
        assert by_version["v2"].n == 20

    def test_single_version_window_backward_compatible(self):
        """Single-version windows still work and produce one VersionPsiResult."""
        records = [
            _make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)
        ]
        data = FakeDriftDataController(records=records)
        store = FakeModelStore()
        poller, em = _make_poller(data=data, store=store)
        poller.poll()

        _, _, version_psi_results = em.emitted[0]
        assert len(version_psi_results) == 1
        assert version_psi_results[0].version == "v1"


# ── Reference cache ────────────────────────────────────────────────────────────


class TestReferenceCache:
    def test_cache_hit_avoids_refetch(self):
        """After the first fetch, same-version subsequent polls must not re-fetch."""
        records = [_make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)]
        data = FakeDriftDataController(records=records)
        store = FakeModelStore()
        poller, _ = _make_poller(
            config=_make_config(reference_cache_ttl_seconds=300),
            data=data,
            store=store,
        )

        poller.poll()
        poller.poll()

        # get_version_artifacts must only have been called once for v1.
        assert store.fetch_counts.get("v1", 0) == 1

    def test_cache_expiry_triggers_refetch(self):
        """A TTL of 0 seconds means every poll re-fetches the reference."""
        records = [_make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)]
        data = FakeDriftDataController(records=records)
        store = FakeModelStore()
        poller, _ = _make_poller(
            config=_make_config(reference_cache_ttl_seconds=0),
            data=data,
            store=store,
        )

        poller.poll()
        poller.poll()

        # With TTL=0 each poll must refresh, so v1 fetched at least twice.
        assert store.fetch_counts.get("v1", 0) >= 2

    def test_stale_cache_used_when_fetch_fails(self):
        """On fetch failure a stale (expired) cache entry is used as fallback."""
        records = [
            _make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)
        ]
        data = FakeDriftDataController(records=records)
        store = FakeModelStore()
        poller, em = _make_poller(
            config=_make_config(reference_cache_ttl_seconds=0),
            data=data,
            store=store,
        )

        # First poll populates cache (TTL=0 means it's already expired after the poll).
        poller.poll()
        assert poller.reference_loaded() is True

        # Second poll: TTL expired → tries to re-fetch, but store now fails.
        store.fail_download = True
        poller.poll()

        _, _, version_psi_results = em.emitted[1]
        # Stale reference should have been used so PSI is not None.
        assert version_psi_results[0].psi is not None

    def test_cache_independent_per_version(self):
        """Each version has its own cache entry; one version expiring does not affect another."""
        records_v1 = [
            _make_record(prediction=c, model_version="v1") for c in range(10) for _ in range(3)
        ]
        records_v2 = [
            _make_record(prediction=c, model_version="v2") for c in range(10) for _ in range(3)
        ]
        data = FakeDriftDataController(records=records_v1 + records_v2)
        store = FakeModelStore()
        poller, _ = _make_poller(
            config=_make_config(reference_cache_ttl_seconds=300),
            data=data,
            store=store,
        )

        poller.poll()
        # Both versions should have been fetched exactly once.
        assert store.fetch_counts.get("v1", 0) == 1
        assert store.fetch_counts.get("v2", 0) == 1

        poller.poll()
        # Within TTL — neither should be re-fetched.
        assert store.fetch_counts.get("v1", 0) == 1
        assert store.fetch_counts.get("v2", 0) == 1


# ── PrometheusEmitter label verification ──────────────────────────────────────


class TestPrometheusEmitterLabels:
    def test_psi_metric_includes_model_version_label(self):
        """The drift_psi_class_distribution metric must carry a model_version label."""
        from monitoring.ml_exporter.main import PrometheusEmitter

        emitter = PrometheusEmitter()
        metrics = WindowMetrics(
            n=30,
            class_counts={c: 3 for c in range(10)},
            class_freqs={c: 0.1 for c in range(10)},
            confidence_mean=0.9,
            psi=None,
        )
        vpr = [VersionPsiResult(version="run-abc", psi=0.05, n=30)]
        emitter.emit(metrics, annotated_count=0, version_psi_results=vpr)

        output = emitter.generate_metrics()[0].decode()
        assert 'drift_psi_class_distribution{model_version="run-abc"}' in output

    def test_version_sample_count_metric_includes_model_version_label(self):
        """drift_window_version_sample_count must carry a model_version label."""
        from monitoring.ml_exporter.main import PrometheusEmitter

        emitter = PrometheusEmitter()
        metrics = WindowMetrics(
            n=30,
            class_counts={c: 3 for c in range(10)},
            class_freqs={c: 0.1 for c in range(10)},
            confidence_mean=0.9,
            psi=None,
        )
        vpr = [VersionPsiResult(version="run-abc", psi=0.05, n=30)]
        emitter.emit(metrics, annotated_count=0, version_psi_results=vpr)

        output = emitter.generate_metrics()[0].decode()
        assert 'drift_window_version_sample_count{model_version="run-abc"}' in output

    def test_multiple_versions_each_get_own_psi_label(self):
        """Each version in version_psi_results must produce a separate labeled series."""
        from monitoring.ml_exporter.main import PrometheusEmitter

        emitter = PrometheusEmitter()
        metrics = WindowMetrics(
            n=60,
            class_counts={c: 6 for c in range(10)},
            class_freqs={c: 0.1 for c in range(10)},
            confidence_mean=0.9,
            psi=None,
        )
        vpr = [
            VersionPsiResult(version="v1", psi=0.05, n=30),
            VersionPsiResult(version="v2", psi=0.30, n=30),
        ]
        emitter.emit(metrics, annotated_count=0, version_psi_results=vpr)

        output = emitter.generate_metrics()[0].decode()
        assert 'drift_psi_class_distribution{model_version="v1"}' in output
        assert 'drift_psi_class_distribution{model_version="v2"}' in output

    def test_sentinel_emitted_for_none_psi(self):
        """VersionPsiResult with psi=None should emit the sentinel value -1."""
        from monitoring.ml_exporter.main import PrometheusEmitter, _PSI_SENTINEL

        emitter = PrometheusEmitter()
        metrics = WindowMetrics(
            n=5,
            class_counts={c: 0 for c in range(10)},
            class_freqs={c: 0.0 for c in range(10)},
            confidence_mean=0.0,
            psi=None,
        )
        vpr = [VersionPsiResult(version="v1", psi=None, n=5)]
        emitter.emit(metrics, annotated_count=0, version_psi_results=vpr)

        output = emitter.generate_metrics()[0].decode()
        assert f'drift_psi_class_distribution{{model_version="v1"}} {_PSI_SENTINEL}' in output
