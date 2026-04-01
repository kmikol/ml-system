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
from shared.model_artifact_controller import ModelArtifactError, ModelStage, ModelStore
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
    model_stage: ModelStage
    poll_interval: int
    window_seconds: int

    @classmethod
    def from_env(cls) -> ExporterConfig:
        return cls(
            model_stage=ModelStage(require_env("MODEL_STAGE")),
            poll_interval=int(require_env("DRIFT_POLL_INTERVAL")),
            window_seconds=int(require_env("DRIFT_WINDOW_SECONDS")),
        )


# ── Layer 2: Pure computation ──────────────────────────────────────────────────


@dataclass
class WindowMetrics:
    n: int
    class_counts: dict[int, int]
    class_freqs: dict[int, float]
    confidence_mean: float
    psi: float | None  # None when reference unavailable or window too small


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
    """Pure function: compute all window metrics from records and optional reference."""
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
    def emit(self, metrics: WindowMetrics, annotated_count: int) -> None: ...
    def update_poll_age(self, age: float) -> None: ...
    def generate_metrics(self) -> tuple[bytes, str]: ...


class PrometheusEmitter:
    """Wraps a private CollectorRegistry so multiple instances can coexist (e.g. in tests)."""

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
            "PSI of prediction class distribution vs. MLflow reference (-1 = insufficient samples)",
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

    def emit(self, metrics: WindowMetrics, annotated_count: int) -> None:
        self._sample_count.set(metrics.n)
        for c in range(10):
            self._class_count.labels(class_label=str(c)).set(metrics.class_counts.get(c, 0))
            self._class_freq.labels(class_label=str(c)).set(metrics.class_freqs.get(c, 0.0))
        self._confidence_mean.set(metrics.confidence_mean)
        self._psi.set(metrics.psi if metrics.psi is not None else _PSI_SENTINEL)
        self._annotated_count.set(annotated_count)

    def update_poll_age(self, age: float) -> None:
        self._poll_age.set(age)

    def generate_metrics(self) -> tuple[bytes, str]:
        return generate_latest(self._registry), CONTENT_TYPE_LATEST


# ── Layer 4: Orchestration ─────────────────────────────────────────────────────


class DriftPoller:
    """Stateful orchestrator: fetches data, manages reference reload, calls emitter.

    All external dependencies are injected via the constructor so the class is
    fully testable without environment variables, a real DB, or MLflow.
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
        self._current_version_id: str | None = None
        self._ref_class_freq: list[float] | None = None
        self._last_poll_ts: float = 0.0

    # ── Public observers ───────────────────────────────────────────────────────

    def current_version_id(self) -> str | None:
        with self._lock:
            return self._current_version_id

    def reference_loaded(self) -> bool:
        with self._lock:
            return self._ref_class_freq is not None

    def poll_age(self) -> float | None:
        """Seconds since the last successful poll, or None if never polled."""
        with self._lock:
            ts = self._last_poll_ts
        return time.time() - ts if ts > 0 else None

    # ── Core poll ──────────────────────────────────────────────────────────────

    def poll(self) -> None:
        """Single poll iteration: check reference, fetch window, emit metrics."""
        self._maybe_reload_reference()

        since = datetime.now(UTC) - timedelta(seconds=self._config.window_seconds)
        try:
            records = self._data.get_predictions(since=since)
        except Exception as e:
            logger.warning(f"DB poll failed: {e}")
            return

        with self._lock:
            ref = self._ref_class_freq

        metrics = compute_window_metrics(records, ref)

        if metrics.psi is None:
            if ref is None:
                logger.warning("Reference distribution not yet loaded, skipping PSI.")
            else:
                logger.info(
                    f"Window too small ({metrics.n} < {_MIN_SAMPLES}), PSI set to sentinel."
                )
        else:
            logger.info(
                f"Poll: n={metrics.n} PSI={metrics.psi:.4f} "
                f"dist={[round(metrics.class_freqs[c], 3) for c in range(10)]}"
            )

        try:
            annotated_count = self._data.get_annotated_count()
        except Exception as e:
            logger.warning(f"Annotated count poll failed: {e}")
            annotated_count = 0

        self._emitter.emit(metrics, annotated_count)

        with self._lock:
            self._last_poll_ts = time.time()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _maybe_reload_reference(self) -> None:
        try:
            version_id = self._store.get_current_version_id(self._config.model_stage)
            with self._lock:
                if version_id != self._current_version_id:
                    logger.info(
                        f"Model version changed: {self._current_version_id} → {version_id}"
                    )
                    ref = self._load_reference()
                    self._current_version_id = version_id
                    self._ref_class_freq = ref
        except ModelArtifactError as e:
            logger.warning(f"Model registry unavailable, using cached reference: {e}")

    def _load_reference(self) -> list[float] | None:
        try:
            ref = self._store.get_reference_distribution(self._config.model_stage)
            freqs = ref.prediction_class_frequencies
            logger.info(f"Reference distribution loaded: {freqs}")
            return freqs
        except Exception as e:
            logger.warning(f"Failed to load reference distribution: {e}")
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
        f"window={config.window_seconds}s stage={config.model_stage.value}"
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
            "model_version_id": poller.current_version_id(),
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
