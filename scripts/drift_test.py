"""
Drift load generator: sends MNIST images with a ramping inversion probability.

Normal MNIST images have a white background with a black digit.
Inverted images (1 - pixel) have a black background with a white digit —
simulating input distribution shift detectable via Mahalanobis score.

Usage:
    python scripts/drift_test.py [--rate RATE] [--duration DURATION]
                                 [--inversion-probability P]
                                 [--ramp SECONDS]
                                 [--url URL]

Inversion probability profile:

    P ────┤              ┌──────────────────────
          │             /
        0 ┼────────────/
          │     ramp

Requests are sent at a constant rate. The fraction of inverted images ramps
linearly from 0 → P over the first `ramp` seconds, then stays flat.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "remaining")
_POOL_SIZE = 1_000  # payloads pre-serialized at startup

HEADERS = {"Content-Type": "application/json"}

_lock = threading.Lock()
_counters: dict[str, int] = defaultdict(int)
_latencies: list[float] = []
_normal_pool: list[bytes] = []
_inverted_pool: list[bytes] = []


def _load_pools() -> tuple[list[bytes], list[bytes]]:
    """Load v0/train MNIST images and pre-serialize normal and inverted pools.

    Images are loaded from the labeled training dataset so that each payload
    includes the original sample UUID as request_id. This ensures prediction_id
    in the predictions table matches dataset_samples.sample_id, which is required
    for the annotation pipeline to look up ground truth labels.
    """
    images_path = os.path.join(_DATA_DIR, "images.npy")
    uuids_path = os.path.join(_DATA_DIR, "uuids.npy")
    try:
        images = np.load(images_path)  # shape (N, 14, 14), float32
        uuids = np.load(uuids_path)    # shape (N,) str
    except FileNotFoundError as e:
        print(f"ERROR: {e}. Run 'make data.prepare' first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(images):,} images. Pre-serializing {_POOL_SIZE:,} payloads each...", flush=True)
    idx = np.random.randint(0, len(images), size=_POOL_SIZE)
    normal = [json.dumps({"image": images[i].tolist(), "request_id": str(uuids[i])}).encode() for i in idx]
    inverted = [json.dumps({"image": (1.0 - images[i]).tolist(), "request_id": str(uuids[i])}).encode() for i in idx]
    return normal, inverted


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    pos = (len(s) - 1) * p / 100
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _inversion_probability_at(t: float, target_p: float, ramp: float) -> float:
    """Inversion probability at time t: ramps linearly from 0 to target_p over ramp seconds."""
    if ramp <= 0:
        return target_p
    return min(target_p * t / ramp, target_p)


def _send(url: str, inv_prob: float) -> None:
    """Send one request; inv_prob is sampled at dispatch time, not inside the thread."""
    if random.random() < inv_prob:
        payload = random.choice(_inverted_pool)
    else:
        payload = random.choice(_normal_pool)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=5.0, help="Req/s (default: 5)")
    parser.add_argument("--duration", type=float, default=120.0, help="Total seconds (default: 120)")
    parser.add_argument(
        "--inversion-probability",
        type=float,
        default=1.0,
        help="Target fraction of inverted images at full ramp (default: 1.0)",
    )
    parser.add_argument(
        "--ramp",
        type=float,
        default=60.0,
        help="Seconds to ramp inversion probability from 0 to target (default: 60)",
    )
    parser.add_argument("--url", default="http://localhost:8000/predict")
    args = parser.parse_args()

    normal, inverted = _load_pools()
    _normal_pool.extend(normal)
    _inverted_pool.extend(inverted)

    total_requests = args.rate * args.duration
    print(f"Rate              : {args.rate} req/s")
    print(f"Duration          : {args.duration}s")
    print(f"Total             : ~{total_requests:.0f} requests")
    print(f"Inversion target  : {args.inversion_probability * 100:.0f}%")
    print(f"Inversion ramp    : {args.ramp}s  (0% → {args.inversion_probability * 100:.0f}% over ramp period)")
    print(f"Target            : {args.url}")
    print("Ctrl-C to stop early\n")

    start = time.perf_counter()
    report_at = start + 5.0
    sent = 0
    interval = 1.0 / args.rate

    while True:
        now = time.perf_counter()
        elapsed = now - start

        if elapsed >= args.duration:
            break

        if now >= report_at:
            with _lock:
                total = _counters["total"]
                ok = _counters.get("status_200", 0)
                lat = list(_latencies)
            avg_ms = sum(lat) / len(lat) if lat else 0
            inv_prob = _inversion_probability_at(elapsed, args.inversion_probability, args.ramp)
            print(
                f"  {elapsed:5.1f}s | inv={inv_prob * 100:4.1f}%  sent={sent:5d} "
                f"completed={total:5d} ok={ok:5d} avg_latency={avg_ms:.0f}ms"
            )
            report_at = now + 5.0

        target_sent = int(elapsed / interval) + 1
        if sent < target_sent:
            inv_prob = _inversion_probability_at(elapsed, args.inversion_probability, args.ramp)
            threading.Thread(target=_send, args=(args.url, inv_prob), daemon=True).start()
            sent += 1
            continue

        ahead_by = sent - (elapsed / interval)
        time.sleep(min(ahead_by * interval, 0.05))

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
