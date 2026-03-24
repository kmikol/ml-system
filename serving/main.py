# serving/main.py
"""
FastAPI inference service. Loads ONNX from MLflow, serves predictions.
Usage: uvicorn serving.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from scipy.special import softmax

from shared.config import require_env
from shared.data_controller.serving import ServingDataController
from shared.logging_config import setup_logging
from shared.model_artifact_controller import ModelArtifactController, ModelArtifactError
from shared.schemas.api import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ValidationErrorResponse,
)
from shared.schemas.predict_record import PredictRecord
from shared.validation import validate_image

setup_logging("serving")
logger = logging.getLogger(__name__)

# ── All config from env, no defaults ─────────────────────────────
MODEL_NAME = require_env("MODEL_NAME")
MODEL_STAGE = require_env("MODEL_STAGE")
POLL_INTERVAL = int(require_env("SERVING_MODEL_POLL_INTERVAL"))
SIMULATED_LATENCY_S = int(require_env("SERVING_SIMULATED_LATENCY_MS")) / 1000.0
# Semaphore caps concurrency to 1. At 333ms service time this saturates at ~3 RPS.
# Requests beyond capacity queue here — latency grows linearly with queue depth.
_concurrency = asyncio.Semaphore(1)

# Arrival counter — incremented before the semaphore so KEDA sees true request
# rate, not throughput. http_requests_total only counts completions and would
# show ~3 RPS regardless of load, preventing scale-out.
_predict_arrivals = Counter("predict_arrivals_total", "Predict requests at arrival")
_prediction_class_counter = Counter(
    "prediction_class_total",
    "Total predictions per class",
    ["class_label"],
)
_confidence_histogram = Histogram(
    "prediction_confidence_score",
    "Confidence score distribution per predicted class",
    ["class_label"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
_mahalanobis_histogram = Histogram(
    "prediction_mahalanobis_score",
    "Squared Mahalanobis distance from predicted class Gaussian",
    ["class_label"],
    buckets=[20, 40, 60, 70, 80, 90, 100, 120, 150, 200, 500],
)

class ModelManager:
    def __init__(self):
        self.classifier_session: ort.InferenceSession | None = None
        self.embedder_session: ort.InferenceSession | None = None
        self.class_gaussians: dict | None = None
        self.model_version: str | None = None
        self._lock = threading.Lock()
        self._artifact_dir = tempfile.mkdtemp(prefix="ml_model_")
        self._controller = ModelArtifactController()

    def load_from_mlflow(self) -> bool:
        try:
            run_id = self._controller.get_production_run_id(MODEL_NAME, MODEL_STAGE)
        except ModelArtifactError as e:
            logger.warning(str(e))
            return False

        if self.model_version == run_id:
            return True

        logger.info(f"Downloading artifacts from run {run_id}...")
        try:
            classifier_path, embedder_path, raw_gaussians = self._controller.download_serving_bundle(
                run_id, self._artifact_dir
            )
        except ModelArtifactError as e:
            logger.error(str(e))
            return False

        logger.info(f"Classifier: {classifier_path}")
        logger.info(f"Embedder:   {embedder_path}")

        # Load ONNX Runtime sessions
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        new_cls = ort.InferenceSession(classifier_path, opts)
        new_emb = ort.InferenceSession(embedder_path, opts)

        new_gaussians = None
        if raw_gaussians is not None:
            try:
                new_gaussians = {
                    cls_key: {
                        "mean": np.array(g["mean"], dtype=np.float64),
                        "precision": np.array(g["precision"], dtype=np.float64),
                    }
                    for cls_key, g in raw_gaussians["classes"].items()
                }
                logger.info("class_gaussians loaded for Mahalanobis scoring.")
            except Exception as e:
                logger.warning(f"class_gaussians payload invalid, Mahalanobis scoring disabled: {e}")
        else:
            logger.info("class_gaussians unavailable, Mahalanobis scoring disabled.")

        with self._lock:
            self.classifier_session = new_cls
            self.embedder_session = new_emb
            self.class_gaussians = new_gaussians
            self.model_version = run_id

        logger.info(f"Model loaded: {run_id}")
        return True

    def predict(self, features_array: np.ndarray) -> dict:
        with self._lock:
            if self.classifier_session is None:
                raise RuntimeError("Model not loaded")
            logits = self.classifier_session.run(
                ["logits"],
                {"features": features_array.astype(np.float32)},
            )[0]
            embedding = self.embedder_session.run(
                ["embedding"],
                {"features": features_array.astype(np.float32)},
            )[0]

        probs = softmax(logits[0])
        prediction = int(np.argmax(probs))
        return {
            "logits": logits[0].tolist(),
            "embedding": embedding[0].tolist(),
            "prediction": prediction,
            "confidence": float(probs[prediction]),
            "prediction_distribution": probs.tolist(),
        }

    @property
    def is_ready(self) -> bool:
        return self.classifier_session is not None


model_manager = ModelManager()
data_controller = ServingDataController()
start_time = time.time()


def poll_model_registry():
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            model_manager.load_from_mlflow()
        except Exception as e:
            logger.error(f"Model poll failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for attempt in range(5):
        if model_manager.load_from_mlflow():
            break
        logger.warning(f"Model load attempt {attempt + 1}/5 failed, retrying in 5s...")
        time.sleep(5)
    else:
        logger.error("Failed to load model after 5 attempts.")

    threading.Thread(target=poll_model_registry, daemon=True).start()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="ML System Serving", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(
    app,
    latency_lowr_buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
).expose(app)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _predict_arrivals.inc()

    errors = validate_image(request.image)
    if errors:
        return JSONResponse(
            status_code=422,
            content=ValidationErrorResponse(detail="Validation failed", errors=errors).model_dump(),
        )

    if SIMULATED_LATENCY_S > 0:
        async with _concurrency:
            await asyncio.sleep(SIMULATED_LATENCY_S)

    features_array = np.array([np.array(request.image).flatten()], dtype=np.float32)
    result = model_manager.predict(features_array)
    request_id = request.request_id or str(uuid.uuid4())
    _prediction_class_counter.labels(class_label=str(result["prediction"])).inc()
    _confidence_histogram.labels(class_label=str(result["prediction"])).observe(result["confidence"])
    gaussians = model_manager.class_gaussians
    if gaussians is not None:
        try:
            g = gaussians[str(result["prediction"])]
            delta = np.array(result["embedding"]) - g["mean"]
            score = float(delta @ g["precision"] @ delta)
            _mahalanobis_histogram.labels(class_label=str(result["prediction"])).observe(score)
        except Exception as e:
            logger.warning(f"Mahalanobis scoring failed: {e}")

    response = PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=model_manager.model_version,
        request_id=request_id,
    )

    data_controller.store_prediction(
        PredictRecord(
            prediction_id=request_id,
            timestamp=datetime.now(UTC),
            model_version=model_manager.model_version,
            image=request.image,
            embedding=result["embedding"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            prediction_distribution=result["prediction_distribution"],
        )
    )

    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    ready = model_manager.is_ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content=HealthResponse(
            status="healthy" if ready else "unhealthy",
            model_loaded=ready,
            model_version=model_manager.model_version,
            uptime_seconds=round(time.time() - start_time, 1),
        ).model_dump(),
    )
