# monitoring/ml_exporter/main.py
"""
ML metrics exporter. Polls the predictions DB on a fixed interval and exposes
domain metrics as Prometheus Gauges: drift (PSI), class distribution, confidence,
and annotation pipeline state.

Design: four layers, no module-level side effects.
  1. ExporterConfig  — pure config object, created in lifespan
  2. Computation     — compute_psi / compute_window_metrics pure functions
  3. MetricsEmitter  — Protocol + PrometheusEmitter (per-instance registry)
  4. DriftPoller     — orchestration; all state as instance attributes

Usage: uvicorn monitoring.ml_exporter.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
import math
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    generate_latest,
)

from shared.config import require_env
from shared.data_controller.drift import DriftDataController
from shared.model_artifact_controller import ModelStore
from shared.schemas.predict_record import PredictRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum predictions in window before PSI is meaningful.
_MIN_SAMPLES = 30
# Sentinel value for PSI when the window is too small or reference unavailable.
_PSI_SENTINEL = -1.0
# Epsilon to prevent log(0) in PSI formula.
_EPS = 1e-6


# ── Layer 1: Config ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExporterConfig:
    """Immutable configuration for the ML exporter.

    Attributes:
        poll_interval: Seconds between poll iterations.
        window_seconds: Width of the sliding prediction window in seconds.
        reference_cache_ttl_seconds: How long (in seconds) a cached per-version
            reference distribution is considered fresh before re-fetching.
    """

    poll_interval: int
    window_seconds: int
    reference_cache_ttl_seconds: int

    @classmethod
    def from_env(cls) -> ExporterConfig:
        """Construct config from environment variables.

        Returns:
            A fully populated ``ExporterConfig``.

        Raises:
            SystemExit: If a required environment variable is missing.
        """
        return cls(
            poll_interval=int(require_env("DRIFT_POLL_INTERVAL")),
            window_seconds=int(require_env("DRIFT_WINDOW_SECONDS")),
            reference_cache_ttl_seconds=int(require_env("REFERENCE_CACHE_TTL_SECONDS")),
        )


# ── Layer 2: Pure computation ──────────────────────────────────────────────────


@dataclass
class WindowMetrics:
    """Aggregate metrics for a prediction window (across all model versions).

    Attributes:
        n: Total number of predictions in the window.
        class_counts: Raw prediction counts per class (0–9).
        class_freqs: Normalised class frequencies per class (0–9).
        confidence_mean: Mean confidence score across all predictions.
        psi: Population Stability Index vs reference, or ``None`` when the
            reference is unavailable or the window is too small.
    """

    n: int
    class_counts: dict[int, int]
    class_freqs: dict[int, float]
    confidence_mean: float
    psi: float | None


@dataclass
class VersionPsiResult:
    """PSI result for a single model version in a poll window.

    Attributes:
        version: The model version identifier (matches ``PredictRecord.model_version``).
        psi: PSI value, or ``None`` when the reference is unavailable or the
            version sub-window has fewer than ``_MIN_SAMPLES`` predictions.
        n: Number of predictions for this version in the current window.
    """

    version: str
    psi: float | None
    n: int


def compute_psi(actual: list[float], reference: list[float]) -> float:
    """
    Compute PSI between actual and reference class distributions.

    PSI = Σ_c (p_actual[c] - p_ref[c]) × ln((p_actual[c] + ε) / (p_ref[c] + ε))

    Thresholds:
      PSI < 0.10  → no significant shift
      PSI < 0.25  → moderate shift
      PSI ≥ 0.25  → significant shift
    """
    psi = 0.0
    for p_a, p_r in zip(actual, reference, strict=True):
        psi += (p_a - p_r) * math.log((p_a + _EPS) / (p_r + _EPS))
    return psi


def compute_window_metrics(
    records: list[PredictRecord],
    reference: list[float] | None,
) -> WindowMetrics:
    """Compute all window metrics from records and an optional reference distribution.

    Args:
        records: Prediction records from the current poll window.
        reference: Normalised reference class frequencies (length 10), or ``None``
            when no reference is available.

    Returns:
        A ``WindowMetrics`` instance.  ``psi`` is ``None`` when *reference* is
        ``None`` or *records* has fewer than ``_MIN_SAMPLES`` entries.
    """
    n = len(records)
    counts: dict[int, int] = {}
    freqs: dict[int, float] = {}

    for c in range(10):
        cnt = sum(1 for r in records if r.prediction == c)
        counts[c] = cnt
        freqs[c] = cnt / n if n > 0 else 0.0

    conf_mean = sum(r.confidence for r in records) / n if n > 0 else 0.0

    psi: float | None = None
    if reference is not None and n >= _MIN_SAMPLES:
        actual_freq = [freqs[c] for c in range(10)]
        psi = compute_psi(actual_freq, reference)

    return WindowMetrics(
        n=n,
        class_counts=counts,
        class_freqs=freqs,
        confidence_mean=conf_mean,
        psi=psi,
    )


# ── Layer 3: Metrics emission ──────────────────────────────────────────────────


class MetricsEmitter(Protocol):
    """Protocol for emitting ML monitoring metrics."""

    def emit(
        self,
        metrics: WindowMetrics,
        annotated_count: int,
        version_psi_results: list[VersionPsiResult],
    ) -> None: ...

    def update_poll_age(self, age: float) -> None: ...

    def generate_metrics(self) -> tuple[bytes, str]: ...


class PrometheusEmitter:
    """Wraps a private CollectorRegistry so multiple instances can coexist (e.g. in tests).

    PSI is emitted with a ``model_version`` label so Grafana can plot per-version
    drift series independently.  A companion ``drift_window_version_sample_count``
    gauge (also labelled by ``model_version``) provides the per-version window size
    for context.
    """

    def __init__(self) -> None:
        self._registry = CollectorRegistry()
        self._sample_count = Gauge(
            "drift_window_sample_count",
            "Predictions in current window",
            registry=self._registry,
        )
        self._class_count = Gauge(
            "drift_window_class_count",
            "Predictions per class in window",
            ["class_label"],
            registry=self._registry,
        )
        self._class_freq = Gauge(
            "drift_window_class_freq",
            "Class proportion in window (0–1)",
            ["class_label"],
            registry=self._registry,
        )
        self._confidence_mean = Gauge(
            "drift_window_confidence_mean",
            "Mean confidence score in window",
            registry=self._registry,
        )
        self._psi = Gauge(
            "drift_psi_class_distribution",
            "PSI of prediction class distribution vs. reference per model version"
            " (-1 = insufficient samples)",
            ["model_version"],
            registry=self._registry,
        )
        self._version_sample_count = Gauge(
            "drift_window_version_sample_count",
            "Predictions in current window for a specific model version",
            ["model_version"],
            registry=self._registry,
        )
        self._poll_age = Gauge(
            "drift_last_poll_age_seconds",
            "Seconds since last successful poll",
            registry=self._registry,
        )
        self._annotated_count = Gauge(
            "annotation_annotated_count",
            "Total predictions with annotation_status='annotated' not yet in any dataset",
            registry=self._registry,
        )

    def emit(
        self,
        metrics: WindowMetrics,
        annotated_count: int,
        version_psi_results: list[VersionPsiResult],
    ) -> None:
        """Publish all window metrics to the Prometheus registry.

        Args:
            metrics: Aggregate window metrics (class counts/freqs, confidence).
            annotated_count: Number of annotated but not-yet-trained predictions.
            version_psi_results: Per-version PSI results to emit with
                ``model_version`` label.
        """
        self._sample_count.set(metrics.n)
        for c in range(10):
            self._class_count.labels(class_label=str(c)).set(metrics.class_counts.get(c, 0))
            self._class_freq.labels(class_label=str(c)).set(metrics.class_freqs.get(c, 0.0))
        self._confidence_mean.set(metrics.confidence_mean)
        for vr in version_psi_results:
            psi_value = vr.psi if vr.psi is not None else _PSI_SENTINEL
            self._psi.labels(model_version=vr.version).set(psi_value)
            self._version_sample_count.labels(model_version=vr.version).set(vr.n)
        self._annotated_count.set(annotated_count)

    def update_poll_age(self, age: float) -> None:
        """Update the poll-age gauge.

        Args:
            age: Seconds since the last successful poll.
        """
        self._poll_age.set(age)

    def generate_metrics(self) -> tuple[bytes, str]:
        """Serialise the registry to Prometheus text format.

        Returns:
            A tuple of ``(payload_bytes, content_type_string)``.
        """
        return generate_latest(self._registry), CONTENT_TYPE_LATEST


# ── Layer 4: Orchestration ─────────────────────────────────────────────────────


class DriftPoller:
    """Stateful orchestrator: fetches data, manages per-version reference cache, calls emitter.

    PSI is computed independently for every model version found in the poll window.
    Reference distributions are cached locally per version with a configurable TTL
    (``ExporterConfig.reference_cache_ttl_seconds``).  On cache miss or expiry the
    distribution is fetched through the ``ModelStore`` facade.

    All external dependencies are injected via the constructor so the class is
    fully testable without environment variables, a real DB, or MLflow.

    Attributes:
        _config: Exporter configuration.
        _data: Data controller for querying predictions.
        _store: Model store facade for fetching reference distributions.
        _emitter: Metrics emitter.
        _lock: Guards all shared mutable state.
        _ref_cache: Per-version reference cache; maps version string to
            ``(class_frequencies, loaded_at_timestamp)``.
        _last_poll_ts: Unix timestamp of the last successful poll.
    """

    def __init__(
        self,
        config: ExporterConfig,
        data_controller: Any,
        store: Any,
        emitter: MetricsEmitter,
    ) -> None:
        self._config = config
        self._data = data_controller
        self._store = store
        self._emitter = emitter

        self._lock = threading.Lock()
        # Maps model_version -> (class_frequencies, loaded_at unix timestamp).
        self._ref_cache: dict[str, tuple[list[float], float]] = {}
        self._last_poll_ts: float = 0.0

    # ── Public observers ───────────────────────────────────────────────────────

    def known_versions(self) -> list[str]:
        """Return the sorted list of model versions with a cached reference.

        Returns:
            Sorted list of version identifier strings, or an empty list if no
            references have been loaded yet.
        """
        with self._lock:
            return sorted(self._ref_cache.keys())

    def reference_loaded(self) -> bool:
        """Return ``True`` if at least one version reference is cached.

        Returns:
            Boolean cache-populated indicator.
        """
        with self._lock:
            return bool(self._ref_cache)

    def poll_age(self) -> float | None:
        """Seconds since the last successful poll, or ``None`` if never polled.

        Returns:
            Elapsed seconds as a float, or ``None``.
        """
        with self._lock:
            ts = self._last_poll_ts
        return time.time() - ts if ts > 0 else None

    # ── Core poll ──────────────────────────────────────────────────────────────

    def poll(self) -> None:
        """Single poll iteration: fetch window, compute per-version PSI, emit metrics.

        Groups records in the current window by ``model_version``, obtains a
        (potentially cached) reference distribution for each version, computes
        PSI independently, and emits all results.  Aggregate window metrics
        (class distribution, confidence) are also emitted for dashboard panels
        that do not require per-version breakdown.
        """
        since = datetime.now(UTC) - timedelta(seconds=self._config.window_seconds)
        try:
            records = self._data.get_predictions(since=since)
        except Exception as e:
            logger.warning(f"DB poll failed: {e}")
            return

        # Group records by model version.
        version_groups: dict[str, list[PredictRecord]] = {}
        for r in records:
            version_groups.setdefault(r.model_version, []).append(r)

        # Compute per-version PSI.
        version_psi_results: list[VersionPsiResult] = []
        for version, vrecords in version_groups.items():
            ref = self._get_reference(version)
            vmetrics = compute_window_metrics(vrecords, ref)
            version_psi_results.append(
                VersionPsiResult(version=version, psi=vmetrics.psi, n=vmetrics.n)
            )
            if vmetrics.psi is None:
                if ref is None:
                    logger.warning(f"Version {version}: reference unavailable, skipping PSI.")
                else:
                    logger.info(
                        f"Version {version}: n={vmetrics.n} < {_MIN_SAMPLES}, PSI set to sentinel."
                    )
            else:
                logger.info(f"Version {version}: n={vmetrics.n} PSI={vmetrics.psi:.4f}")

        # Aggregate metrics (class freq, confidence, total count) across all versions.
        aggregate_metrics = compute_window_metrics(records, reference=None)

        try:
            annotated_count = self._data.get_annotated_count()
        except Exception as e:
            logger.warning(f"Annotated count poll failed: {e}")
            annotated_count = 0

        self._emitter.emit(aggregate_metrics, annotated_count, version_psi_results)

        with self._lock:
            self._last_poll_ts = time.time()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_reference(self, version: str) -> list[float] | None:
        """Return cached reference class frequencies for *version*, fetching if needed.

        Cache entries are considered fresh for ``reference_cache_ttl_seconds``
        seconds.  On a cache miss or expiry the reference is fetched through the
        ``ModelStore`` facade via ``get_version_artifacts``.  If fetching fails and
        a stale entry exists it is returned as a fallback to avoid a gap in PSI
        monitoring.

        Args:
            version: The model version identifier (matches ``PredictRecord.model_version``).

        Returns:
            Normalised class frequency list (length 10), or ``None`` when no
            reference is available (cache empty and fetch failed).
        """
        now = time.time()
        with self._lock:
            cached = self._ref_cache.get(version)

        if cached is not None:
            freqs, loaded_at = cached
            if now - loaded_at < self._config.reference_cache_ttl_seconds:
                return freqs

        # Cache miss or TTL expired — fetch through the model store facade.
        try:
            artifacts = self._store.get_version_artifacts(version, include_reference=True)
            if artifacts.reference_distribution is None:
                logger.warning(f"No reference distribution for version {version}.")
                return None
            freqs = artifacts.reference_distribution.prediction_class_frequencies
            logger.info(f"Reference loaded for version {version}: {freqs}")
            with self._lock:
                self._ref_cache[version] = (freqs, time.time())
            return freqs
        except Exception as e:
            logger.warning(f"Failed to load reference for version {version}: {e}")
            # Return stale cache entry rather than dropping PSI entirely.
            with self._lock:
                stale = self._ref_cache.get(version)
            if stale is not None:
                logger.info(f"Using stale reference for version {version}.")
                return stale[0]
            return None


# ── Background threads ─────────────────────────────────────────────────────────


def _poll_loop(poller: DriftPoller, config: ExporterConfig) -> None:
    while True:
        try:
            poller.poll()
        except Exception as e:
            logger.error(f"Unexpected error in poll loop: {e}")
        time.sleep(config.poll_interval)


def _age_updater(poller: DriftPoller, emitter: MetricsEmitter) -> None:
    """Continuously update the poll-age gauge so staleness is visible in Grafana."""
    while True:
        age = poller.poll_age()
        emitter.update_poll_age(age if age is not None else float("inf"))
        time.sleep(1)


# ── FastAPI app ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ExporterConfig.from_env()
    emitter = PrometheusEmitter()
    poller = DriftPoller(
        config=config,
        data_controller=DriftDataController(),
        store=ModelStore(),
        emitter=emitter,
    )

    app.state.poller = poller
    app.state.emitter = emitter
    app.state.config = config

    threading.Thread(target=_poll_loop, args=(poller, config), daemon=True).start()
    threading.Thread(target=_age_updater, args=(poller, emitter), daemon=True).start()

    logger.info(
        f"ML exporter started — poll_interval={config.poll_interval}s "
        f"window={config.window_seconds}s"
    )
    yield
    logger.info("ML exporter shutting down.")


app = FastAPI(title="ML Exporter", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    poller: DriftPoller = request.app.state.poller
    config: ExporterConfig = request.app.state.config
    age = poller.poll_age()
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "known_model_versions": poller.known_versions(),
            "reference_loaded": poller.reference_loaded(),
            "last_poll_age_seconds": round(age, 1) if age is not None else None,
            "poll_interval": config.poll_interval,
            "window_seconds": config.window_seconds,
        },
    )


@app.get("/metrics")
async def metrics_endpoint(request: Request):
    emitter: PrometheusEmitter = request.app.state.emitter
    data, content_type = emitter.generate_metrics()
    return Response(content=data, media_type=content_type)
