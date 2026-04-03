"""
Variable-rate load generator for the serving endpoint.

Usage:
    python scripts/load_test.py [--rate RATE] [--duration DURATION]
                                [--ramp-up SECONDS] [--ramp-down SECONDS]
                                [--url URL]

Rate profile:

    rate ─┤         ┌───────────────────┐
          │        /                     \\
        0 ┼───────/                       \\──────
          │ ramp-up     steady state    ramp-down

Sends are driven by integrating the rate curve, so the actual send count
always tracks the expected count regardless of how slow/fast the ramp is.

Defaults: 5 req/s, 60s, no ramp.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
from collections import defaultdict

import numpy as np

_POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "remaining", "images.npy")
_UUIDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "remaining", "uuids.npy")
_POOL_SIZE = 1_000  # payloads pre-serialized at startup; avoids per-request numpy/json overhead

HEADERS = {"Content-Type": "application/json"}

_lock = threading.Lock()
_counters: dict[str, int] = defaultdict(int)
_latencies: list[float] = []  # per-request ms, collected under _lock
_pool: list[bytes] = []  # populated in main() before any threads start


def _load_pool() -> list[bytes]:
    """Load remaining MNIST images and pre-serialize a random sample to JSON bytes."""
    try:
        images = np.load(_POOL_PATH)  # shape (N, 14, 14), float32
        uuids = np.load(_UUIDS_PATH)  # shape (N,), str — assigned at prepare time
    except FileNotFoundError as e:
        print(f"ERROR: {e}. Run 'make data.prepare' first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(images):,} images. Pre-serializing {_POOL_SIZE:,} payloads...", flush=True)
    idx = np.random.randint(0, len(images), size=_POOL_SIZE)
    return [json.dumps({"image": images[i].tolist(), "uuid": str(uuids[i])}).encode() for i in idx]


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    pos = (len(s) - 1) * p / 100
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _detect_default_url() -> str:
    candidates = [
        ("http://localhost/serving", "http://localhost/serving/health"),
        ("http://localhost:8000", "http://localhost:8000/health"),
    ]
    for base, health in candidates:
        try:
            with urllib.request.urlopen(health, timeout=2) as resp:
                if resp.status == 200:
                    return urllib.parse.urljoin(base + "/", "predict")
        except Exception:
            continue
    return "http://localhost/serving/predict"


def _send(url: str) -> None:
    payload = random.choice(_pool)
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=payload, headers=HEADERS, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
    except urllib.error.URLError as e:
        print(f"  Connection error: {e.reason}", file=sys.stderr)
        status = 0
    except Exception as e:
        print(f"  Unexpected error ({type(e).__name__}): {e}", file=sys.stderr)
        status = 0
    elapsed_ms = (time.perf_counter() - t0) * 1000
    with _lock:
        _counters["total"] += 1
        _counters[f"status_{status}"] += 1
        _latencies.append(elapsed_ms)


def _current_rate(
    t: float, peak: float, ramp_up: float, duration: float, ramp_down: float
) -> float:
    """Instantaneous rate at time t."""
    if ramp_up > 0 and t < ramp_up:
        return peak * t / ramp_up
    if ramp_down > 0 and t > duration - ramp_down:
        return peak * max(duration - t, 0.0) / ramp_down
    return peak


def _total_expected(
    t: float, peak: float, ramp_up: float, duration: float, ramp_down: float
) -> float:
    """
    Integral of the rate curve from 0 to t.
    Gives the cumulative number of requests that should have been sent by time t.
    Driving sends from this value avoids the 1/rate tick problem at low rates.
    """
    t = max(0.0, min(t, duration))
    steady_start = ramp_up
    steady_end = duration - ramp_down
    result = 0.0

    # Phase 1: ramp-up [0, ramp_up)
    if ramp_up > 0 and t > 0:
        t_in = min(t, ramp_up)
        result += peak * t_in**2 / (2 * ramp_up)

    # Phase 2: steady [ramp_up, duration - ramp_down)
    if t > steady_start:
        t_in = min(t, steady_end) - steady_start
        if t_in > 0:
            result += peak * t_in

    # Phase 3: ramp-down [duration - ramp_down, duration)
    if ramp_down > 0 and t > steady_end:
        t_in = t - steady_end
        result += peak * t_in - peak * t_in**2 / (2 * ramp_down)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=5.0, help="Peak req/s (default: 5)")
    parser.add_argument("--duration", type=float, default=60.0, help="Total seconds (default: 60)")
    parser.add_argument("--ramp-up", type=float, default=0.0, help="Ramp-up seconds (default: 0)")
    parser.add_argument(
        "--ramp-down", type=float, default=0.0, help="Ramp-down seconds (default: 0)"
    )
    parser.add_argument("--url", default=None)
    args = parser.parse_args()
    url = args.url or _detect_default_url()

    _pool.extend(_load_pool())

    total_expected_final = _total_expected(
        args.duration, args.rate, args.ramp_up, args.duration, args.ramp_down
    )
    print(f"Peak rate   : {args.rate} req/s")
    print(
        f"Duration    : {args.duration}s  "
        f"(ramp-up {args.ramp_up}s / steady / ramp-down {args.ramp_down}s)"
    )
    print(f"Total       : ~{total_expected_final:.0f} requests")
    print(f"Target      : {url}")
    print("Ctrl-C to stop early\n")

    start = time.perf_counter()
    report_at = start + 5.0
    sent = 0

    while True:
        now = time.perf_counter()
        elapsed = now - start

        if elapsed >= args.duration:
            break

        # Progress report — always checked first so ramp doesn't delay it.
        if now >= report_at:
            with _lock:
                total = _counters["total"]
                ok = _counters.get("status_200", 0)
                lat = list(_latencies)
            avg_ms = sum(lat) / len(lat) if lat else 0
            rate = _current_rate(elapsed, args.rate, args.ramp_up, args.duration, args.ramp_down)
            print(
                f"  {elapsed:5.1f}s | rate={rate:5.1f} sent={sent:5d} "
                f"completed={total:5d} ok={ok:5d} avg_latency={avg_ms:.0f}ms"
            )
            report_at = now + 5.0

        # Fire if we're behind the expected cumulative count.
        target = _total_expected(elapsed, args.rate, args.ramp_up, args.duration, args.ramp_down)
        if sent < target:
            threading.Thread(target=_send, args=(url,), daemon=True).start()
            sent += 1
            continue  # re-check immediately in case we're still behind

        # Ahead or on schedule — sleep until approximately the next send.
        rate = _current_rate(elapsed, args.rate, args.ramp_up, args.duration, args.ramp_down)
        if rate > 0:
            ahead_by = sent - target  # requests ahead of schedule
            wait = ahead_by / rate  # seconds until schedule catches up
            time.sleep(min(wait, 0.05))
        else:
            time.sleep(0.05)

    time.sleep(2)  # let in-flight requests finish

    elapsed = time.perf_counter() - start
    with _lock:
        total = _counters["total"]
        ok = _counters.get("status_200", 0)
        lat = list(_latencies)
        statuses = {k: v for k, v in _counters.items() if k.startswith("status_")}

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Sent        : {sent}")
    print(f"  Completed   : {total}")
    print(f"  200 OK      : {ok}")
    if lat:
        print(f"  Latency avg : {sum(lat) / len(lat):.0f} ms")
        print(f"  Latency p50 : {_percentile(lat, 50):.0f} ms")
        print(f"  Latency p95 : {_percentile(lat, 95):.0f} ms")
        print(f"  Latency p99 : {_percentile(lat, 99):.0f} ms")
    print(f"  By status   : {statuses}")


if __name__ == "__main__":
    main()
