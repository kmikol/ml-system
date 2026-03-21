# serving/main.py
"""
FastAPI inference service. Loads ONNX from MLflow, serves predictions.
Usage: uvicorn serving.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import time
import os
import uuid
import logging
import threading
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from shared.config import require_env
from shared.model_artifact_controller import MLflowModelArtifactController, ModelArtifactError
from shared.artifact_paths import (
    MLFLOW_PATH_ONNX_ROOT,
    resolve_classifier_path,
    resolve_embedder_path,
)
from shared.schemas.api import (
    PredictRequest, PredictResponse, HealthResponse, ValidationErrorResponse,
)
from shared.schemas.feature_schema import FEATURE_NAMES, INPUT_DIM
from shared.validation import validate_features

logging.basicConfig(level=logging.INFO)
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

REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_STREAM_NAME = os.getenv("REDIS_STREAM_NAME", "inference_events")


class ModelManager:
    def __init__(self):
        self.classifier_session: ort.InferenceSession | None = None
        self.embedder_session: ort.InferenceSession | None = None
        self.model_version: str | None = None
        self._lock = threading.Lock()
        self._artifact_dir = tempfile.mkdtemp(prefix="ml_model_")
        self._controller = MLflowModelArtifactController()

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
            onnx_dir = self._controller.download_artifacts(
                run_id, MLFLOW_PATH_ONNX_ROOT, self._artifact_dir
            )
        except ModelArtifactError as e:
            logger.error(str(e))
            return False

        logger.info(f"Downloaded to: {onnx_dir}")

        # Resolve paths (crashes with diagnostic dump if files missing)
        classifier_path = resolve_classifier_path(onnx_dir)
        embedder_path = resolve_embedder_path(onnx_dir)
        logger.info(f"Classifier: {classifier_path}")
        logger.info(f"Embedder:   {embedder_path}")

        # Load ONNX Runtime sessions
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        new_cls = ort.InferenceSession(classifier_path, opts)
        new_emb = ort.InferenceSession(embedder_path, opts)

        with self._lock:
            self.classifier_session = new_cls
            self.embedder_session = new_emb
            self.model_version = run_id

        logger.info(f"Model loaded: {run_id}")
        return True

    def predict(self, features_array: np.ndarray) -> dict:
        with self._lock:
            if self.classifier_session is None:
                raise RuntimeError("Model not loaded")
            logits = self.classifier_session.run(
                ["logits"], {"features": features_array.astype(np.float32)},
            )[0]
            embedding = self.embedder_session.run(
                ["embedding"], {"features": features_array.astype(np.float32)},
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


class RedisPublisher:
    def __init__(self):
        self._client = None
        self._available = False
        self._failures = 0

    def connect(self):
        try:
            import redis
            self._client = redis.from_url(REDIS_URL, socket_timeout=1)
            self._client.ping()
            self._available = True
            logger.info(f"Redis connected: {REDIS_URL}")
        except Exception as e:
            logger.warning(f"Redis unavailable (serving continues without): {e}")
            self._available = False

    def publish(self, event_json: str):
        if not self._available:
            return
        try:
            self._client.xadd(REDIS_STREAM_NAME, {"payload": event_json}, maxlen=100000, approximate=True)
        except Exception as e:
            self._failures += 1
            logger.warning(f"Redis publish failed ({self._failures} total): {e}")


model_manager = ModelManager()
redis_publisher = RedisPublisher()
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

    redis_publisher.connect()
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

    errors = validate_features(request.features)
    if errors:
        return JSONResponse(
            status_code=422,
            content=ValidationErrorResponse(detail="Validation failed", errors=errors).model_dump(),
        )

    if SIMULATED_LATENCY_S > 0:
        async with _concurrency:
            await asyncio.sleep(SIMULATED_LATENCY_S)

    features_array = np.array([[request.features[n] for n in FEATURE_NAMES]], dtype=np.float32)
    result = model_manager.predict(features_array)
    request_id = request.request_id or str(uuid.uuid4())

    response = PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=model_manager.model_version,
        request_id=request_id,
    )

    try:
        from shared.schemas.inference_event import InferenceEvent
        event = InferenceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            model_version=model_manager.model_version,
            request_id=request_id,
            features=request.features,
            embedding=result["embedding"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            prediction_distribution=result["prediction_distribution"],
        )
        redis_publisher.publish(event.model_dump_json())
    except Exception as e:
        logger.warning(f"Redis publish failed: {e}")

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
