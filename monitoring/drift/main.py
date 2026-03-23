# monitoring/drift/main.py
"""
Drift detection service. Polls the predictions DB on a fixed interval,
computes PSI over the prediction class distribution vs. the MLflow reference
baseline, and exposes results as Prometheus Gauges.

Usage: uvicorn monitoring.drift.main:app --host 0.0.0.0 --port 8001
"""

import logging
import math
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from collections import Counter
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import Gauge, make_asgi_app

from shared.artifact_paths import REFERENCE_DIST_FILENAME
from shared.config import require_env
from shared.data_controller.drift import DriftDataController
from shared.model_artifact_controller import ModelArtifactError
from shared.model_artifact_controller.mlflow import MLflowModelArtifactController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
MODEL_NAME = require_env("MODEL_NAME")
MODEL_STAGE = require_env("MODEL_STAGE")
DRIFT_POLL_INTERVAL = int(require_env("DRIFT_POLL_INTERVAL"))
DRIFT_WINDOW_SECONDS = int(require_env("DRIFT_WINDOW_SECONDS"))

# Minimum predictions in window before PSI is meaningful.
_MIN_SAMPLES = 30
# Sentinel value for PSI when the window is too small.
_PSI_SENTINEL = -1.0
# Epsilon to prevent log(0) in PSI formula.
_EPS = 1e-6

# ── Prometheus metrics ────────────────────────────────────────────
_sample_count = Gauge("drift_window_sample_count", "Predictions in current window")
_class_count = Gauge(
    "drift_window_class_count", "Predictions per class in window", ["class_label"]
)
_class_freq = Gauge(
    "drift_window_class_freq", "Class proportion in window (0–1)", ["class_label"]
)
_confidence_mean = Gauge("drift_window_confidence_mean", "Mean confidence score in window")
_psi = Gauge(
    "drift_psi_class_distribution",
    "PSI of prediction class distribution vs. MLflow reference "
    "(-1 = insufficient samples)",
)
_poll_age = Gauge(
    "drift_last_poll_age_seconds", "Seconds since last successful poll"
)


# ── Shared state (read/written by poll thread, read by health endpoint) ──
_lock = threading.Lock()
_current_run_id: str | None = None
_ref_class_freq: list[float] | None = None  # 10-element list from reference_distribution.json
_last_poll_ts: float = 0.0
_artifact_dir = tempfile.mkdtemp(prefix="ml_drift_")

_data_controller = DriftDataController()
_artifact_controller = MLflowModelArtifactController()


def _load_reference(run_id: str) -> list[float] | None:
    """Download reference_distribution.json and return prediction_class_frequencies."""
    try:
        path = _artifact_controller.download_artifacts(run_id, REFERENCE_DIST_FILENAME, _artifact_dir)
        import json
        with open(path) as f:
            data = json.load(f)
        freqs = data["prediction_class_frequencies"]
        logger.info(f"Reference distribution loaded from run {run_id}: {freqs}")
        return freqs
    except Exception as e:
        logger.warning(f"Failed to load reference distribution from run {run_id}: {e}")
        return None


def _compute_psi(actual: list[float], reference: list[float]) -> float:
    """
    Compute PSI between actual and reference class distributions.

    PSI = Σ_c (p_actual[c] - p_ref[c]) × ln((p_actual[c] + ε) / (p_ref[c] + ε))

    Thresholds:
      PSI < 0.10  → no significant shift
      PSI < 0.25  → moderate shift
      PSI ≥ 0.25  → significant shift
    """
    psi = 0.0
    for p_a, p_r in zip(actual, reference):
        psi += (p_a - p_r) * math.log((p_a + _EPS) / (p_r + _EPS))
    return psi


def _poll() -> None:
    """Single poll: fetch window, compute metrics, update Gauges."""
    global _current_run_id, _ref_class_freq, _last_poll_ts

    # Check for model version change and reload reference if needed.
    try:
        run_id = _artifact_controller.get_production_run_id(MODEL_NAME, MODEL_STAGE)
        with _lock:
            if run_id != _current_run_id:
                logger.info(f"Model version changed: {_current_run_id} → {run_id}")
                ref = _load_reference(run_id)
                _current_run_id = run_id
                _ref_class_freq = ref
    except ModelArtifactError as e:
        logger.warning(f"MLflow unavailable, using cached reference: {e}")

    # Fetch predictions window.
    since = datetime.now(UTC) - timedelta(seconds=DRIFT_WINDOW_SECONDS)
    try:
        records = _data_controller.get_predictions(since=since)
    except Exception as e:
        logger.warning(f"DB poll failed: {e}")
        return

    n = len(records)
    _sample_count.set(n)

    if n == 0:
        for c in range(10):
            _class_count.labels(class_label=str(c)).set(0)
            _class_freq.labels(class_label=str(c)).set(0.0)
        _confidence_mean.set(0.0)
        _psi.set(_PSI_SENTINEL)
        with _lock:
            _last_poll_ts = time.time()
        return

    # Class distribution.
    counts = Counter(r.prediction for r in records)
    actual_freq = []
    for c in range(10):
        cnt = counts.get(c, 0)
        freq = cnt / n
        _class_count.labels(class_label=str(c)).set(cnt)
        _class_freq.labels(class_label=str(c)).set(freq)
        actual_freq.append(freq)

    # Mean confidence.
    _confidence_mean.set(sum(r.confidence for r in records) / n)

    # PSI.
    with _lock:
        ref = _ref_class_freq

    if ref is None:
        logger.warning("Reference distribution not yet loaded, skipping PSI.")
        _psi.set(_PSI_SENTINEL)
    elif n < _MIN_SAMPLES:
        logger.info(f"Window too small ({n} < {_MIN_SAMPLES}), PSI set to sentinel.")
        _psi.set(_PSI_SENTINEL)
    else:
        psi_val = _compute_psi(actual_freq, ref)
        _psi.set(psi_val)
        logger.info(
            f"Poll: n={n} PSI={psi_val:.4f} "
            f"dist={[round(f, 3) for f in actual_freq]}"
        )

    with _lock:
        _last_poll_ts = time.time()


def _poll_loop() -> None:
    while True:
        try:
            _poll()
        except Exception as e:
            logger.error(f"Unexpected error in poll loop: {e}")
        time.sleep(DRIFT_POLL_INTERVAL)


def _age_updater() -> None:
    """Continuously update the poll-age gauge so staleness is visible in Grafana."""
    while True:
        with _lock:
            ts = _last_poll_ts
        age = time.time() - ts if ts > 0 else float("inf")
        _poll_age.set(age)
        time.sleep(1)


# ── FastAPI app ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_poll_loop, daemon=True).start()
    threading.Thread(target=_age_updater, daemon=True).start()
    logger.info(
        f"Drift service started — poll_interval={DRIFT_POLL_INTERVAL}s "
        f"window={DRIFT_WINDOW_SECONDS}s model={MODEL_NAME}/{MODEL_STAGE}"
    )
    yield
    logger.info("Drift service shutting down.")


app = FastAPI(title="Drift Detection", version="1.0.0", lifespan=lifespan)

# Mount prometheus_client's ASGI app at /metrics (no fastapi-instrumentator needed).
app.mount("/metrics", make_asgi_app())


@app.get("/health")
async def health():
    with _lock:
        run_id = _current_run_id
        ref_loaded = _ref_class_freq is not None
        ts = _last_poll_ts
    age = time.time() - ts if ts > 0 else None
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "model_run_id": run_id,
            "reference_loaded": ref_loaded,
            "last_poll_age_seconds": round(age, 1) if age is not None else None,
            "poll_interval": DRIFT_POLL_INTERVAL,
            "window_seconds": DRIFT_WINDOW_SECONDS,
        },
    )
