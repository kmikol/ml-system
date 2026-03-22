"""Integration tests — requires: make infra.up && make train && make serve"""

import httpx

BASE = "http://localhost:8000"
VALID = {
    "age": 35.0,
    "income": 55000.0,
    "credit_score": 720.0,
    "debt_ratio": 1.2,
    "num_accounts": 5.0,
}


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict():
    r = httpx.post(f"{BASE}/predict", json={"features": VALID}, timeout=10)
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["confidence"] <= 1
    assert d["prediction"] in [0, 1, 2]


def test_predict_bad_input():
    r = httpx.post(f"{BASE}/predict", json={"features": {"age": 5.0}}, timeout=10)
    assert r.status_code == 422
