"""Integration tests for the serving API.

Requires a running serving instance with a trained model:
  make k3d.redeploy  (or make serve via docker compose)

Endpoint: http://localhost:8000
"""

import httpx
import pytest

BASE = "http://localhost:8000"

# Blank 14x14 image (all zeros — valid, in-distribution for MNIST background)
_BLANK = [[0.0] * 14 for _ in range(14)]

# White 14x14 image (all ones — valid pixel values)
_WHITE = [[1.0] * 14 for _ in range(14)]


def test_health_returns_200_when_model_loaded():
    r = httpx.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "healthy"
    assert body["model_version"] is not None
    assert body["uptime_seconds"] >= 0


def test_predict_blank_image_returns_valid_response():
    r = httpx.post(f"{BASE}/predict", json={"image": _BLANK}, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in range(10)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] is not None
    assert isinstance(body["uuid"], str)


def test_predict_white_image_returns_valid_response():
    r = httpx.post(f"{BASE}/predict", json={"image": _WHITE}, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in range(10)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_propagates_uuid():
    import uuid
    sample_uuid = str(uuid.uuid4())
    r = httpx.post(
        f"{BASE}/predict",
        json={"image": _BLANK, "uuid": sample_uuid},
        timeout=10,
    )
    assert r.status_code == 200
    assert r.json()["uuid"] == sample_uuid


def test_predict_generates_uuid_when_omitted():
    r = httpx.post(f"{BASE}/predict", json={"image": _BLANK}, timeout=10)
    assert r.status_code == 200
    assert len(r.json()["uuid"]) > 0


@pytest.mark.parametrize("bad_body,description", [
    ({"image": [[0.0] * 14 for _ in range(13)]}, "13 rows instead of 14"),
    ({"image": [[0.0] * 13 for _ in range(14)]}, "13 cols instead of 14"),
    ({"image": [[-0.1] * 14 for _ in range(14)]}, "pixel value below 0"),
    ({"image": [[1.1] * 14 for _ in range(14)]}, "pixel value above 1"),
    ({}, "missing image field"),
])
def test_predict_rejects_invalid_input(bad_body, description):
    r = httpx.post(f"{BASE}/predict", json=bad_body, timeout=10)
    assert r.status_code == 422, f"Expected 422 for: {description}"
