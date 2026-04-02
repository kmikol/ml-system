"""End-to-end integration tests for the serving container.

Exercises the full request path: HTTP → ONNX inference → Postgres → MinIO.
Unit tests mock the data controller, so this is the only layer that catches
breaks in the actual persistence path.

Requires a running serving instance connected to real Postgres and MinIO:
  docker-compose -f docker-compose.test.yml up serving

Environment variables (set by docker-compose.test.yml):
  SERVING_BASE_URL          - defaults to http://localhost:8000
  DATA_CONTROLLER_DB_URL    - Postgres DSN
  DATASET_S3_ENDPOINT_URL   - MinIO endpoint
  DATASET_BUCKET            - bucket where prediction images are stored
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
  MLFLOW_TRACKING_URI       - required only for test_model_hot_swap
  MODEL_NAME                - registered model name
  SERVING_MODEL_POLL_INTERVAL - used to time the hot-swap wait
"""

from __future__ import annotations

import os
import time
import uuid

import boto3
import httpx
import psycopg2
import pytest

BASE = os.environ.get("SERVING_BASE_URL", "http://localhost:8000")
DB_URL = os.environ.get("DATA_CONTROLLER_DB_URL", "")
S3_ENDPOINT = os.environ.get("DATASET_S3_ENDPOINT_URL", "")
DATASET_BUCKET = os.environ.get("DATASET_BUCKET", "test-dataset")
AWS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "ml_system_model")
POLL_INTERVAL = int(os.environ.get("SERVING_MODEL_POLL_INTERVAL", "5"))

_BLANK = [[0.0] * 14 for _ in range(14)]


@pytest.fixture(scope="module")
def db():
    if not DB_URL:
        pytest.skip("DATA_CONTROLLER_DB_URL not set")
    conn = psycopg2.connect(DB_URL)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def s3():
    if not S3_ENDPOINT:
        pytest.skip("DATASET_S3_ENDPOINT_URL not set")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )


# ── basic health and inference ────────────────────────────────────────────────


def test_health_returns_ready():
    r = httpx.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True, "Serving reports model not loaded"
    assert body["status"] == "healthy"
    assert body["model_version"] is not None


def test_predict_returns_valid_response():
    r = httpx.post(f"{BASE}/predict", json={"image": _BLANK}, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in range(10)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] is not None
    assert body["uuid"]


# ── full persistence path ─────────────────────────────────────────────────────


def test_prediction_persisted_in_postgres(db):
    """POST /predict must write a row to the predictions table.

    This test catches breaks where the serving container starts and responds
    correctly but silently fails to write to Postgres (e.g. wrong DB_URL,
    missing table, schema mismatch).
    """
    sample_uuid = str(uuid.uuid4())
    r = httpx.post(f"{BASE}/predict", json={"image": _BLANK, "uuid": sample_uuid}, timeout=10)
    assert r.status_code == 200

    with db.cursor() as cur:
        cur.execute(
            "SELECT uuid, annotation_status FROM predictions WHERE uuid = %s",
            (sample_uuid,),
        )
        row = cur.fetchone()

    assert row is not None, (
        f"No row in predictions table for uuid {sample_uuid}. "
        "Check DATA_CONTROLLER_DB_URL and that the predictions table exists."
    )
    assert str(row[0]) == sample_uuid
    assert row[1] == "none", f"Expected annotation_status='none', got '{row[1]}'"


def test_prediction_image_stored_in_minio(s3):
    """POST /predict must upload the input image to MinIO at predictions/{uuid}.npy.

    This test catches breaks where inference succeeds but the image upload
    silently fails (e.g. wrong bucket, bad credentials, missing DATASET_BUCKET).
    """
    sample_uuid = str(uuid.uuid4())
    r = httpx.post(f"{BASE}/predict", json={"image": _BLANK, "uuid": sample_uuid}, timeout=10)
    assert r.status_code == 200

    expected_key = f"predictions/{sample_uuid}.npy"
    response = s3.list_objects_v2(Bucket=DATASET_BUCKET, Prefix=f"predictions/{sample_uuid}")
    objects = response.get("Contents", [])
    keys = [o["Key"] for o in objects]

    assert expected_key in keys, (
        f"Expected MinIO object '{expected_key}' not found in bucket '{DATASET_BUCKET}'. "
        f"Objects with matching prefix: {keys}"
    )


# ── input validation ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_body,description",
    [
        ({"image": [[0.0] * 14 for _ in range(13)]}, "13 rows instead of 14"),
        ({"image": [[0.0] * 13 for _ in range(14)]}, "13 cols instead of 14"),
        ({"image": [[-0.1] * 14 for _ in range(14)]}, "pixel below 0"),
        ({"image": [[1.1] * 14 for _ in range(14)]}, "pixel above 1"),
        ({}, "missing image field"),
    ],
)
def test_predict_rejects_invalid_input(bad_body, description):
    r = httpx.post(f"{BASE}/predict", json=bad_body, timeout=10)
    assert r.status_code == 422, f"Expected 422 for: {description}, got {r.status_code}"


# ── model hot-swap ────────────────────────────────────────────────────────────


def test_model_hot_swap():
    """Serving must pick up a newly promoted model without a pod restart.

    Registers a second model version in MLflow, promotes it to Production,
    then waits for the serving polling thread to detect the change. This tests
    the ModelManager background thread and its interaction with a real MLflow.
    """
    if not MLFLOW_URI:
        pytest.skip("MLFLOW_TRACKING_URI not set — cannot register a second model version")

    initial_version = httpx.get(f"{BASE}/health", timeout=10).json()["model_version"]
    assert initial_version is not None, "Serving is not ready — model not loaded"

    from shared.model_artifact_controller.mlflow import MLflowModelArtifactController
    from tests.integration.seed_model import seed_one_model

    controller = MLflowModelArtifactController()
    new_version = seed_one_model(MODEL_NAME, controller)

    # The serving polling thread wakes every POLL_INTERVAL seconds.
    # Allow an extra buffer so a single missed poll window doesn't flake.
    wait_seconds = POLL_INTERVAL + 10
    time.sleep(wait_seconds)

    updated_version = httpx.get(f"{BASE}/health", timeout=10).json()["model_version"]
    assert updated_version != initial_version, (
        f"model_version did not change after {wait_seconds}s.\n"
        f"Initial: {initial_version}, current: {updated_version}, new seed: {new_version}.\n"
        f"Check SERVING_MODEL_POLL_INTERVAL={POLL_INTERVAL} and MLflow connectivity."
    )
