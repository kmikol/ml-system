"""
Fixed-rate load generator for the serving endpoint.

Usage:
    python scripts/load_test.py [--rate RATE] [--duration DURATION] [--url URL]

Defaults: 5 req/s for 60 s against http://localhost:8000/predict
"""
import argparse
import time
import urllib.request
import urllib.error
import json
import threading
from collections import defaultdict

PAYLOAD = json.dumps({
    "features": {
        "age": 35.0,
        "income": 55000.0,
        "credit_score": 720.0,
        "debt_ratio": 1.2,
        "num_accounts": 5.0,
    }
}).encode()

HEADERS = {"Content-Type": "application/json"}

_lock = threading.Lock()
_counters: dict[str, int] = defaultdict(int)


def _send(url: str) -> None:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=PAYLOAD, headers=HEADERS, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
            resp.read()  # consume body so timer includes full round-trip, not just headers
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception:
        status = 0
    elapsed = time.perf_counter() - t0
    with _lock:
        _counters["total"] += 1
        _counters[f"status_{status}"] += 1
        _counters["latency_sum_ms"] += int(elapsed * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-rate load generator")
    parser.add_argument("--rate", type=float, default=5.0, help="Requests per second (default: 5)")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (default: 60)")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="Target URL")
    args = parser.parse_args()

    interval = 1.0 / args.rate
    total_requests = int(args.rate * args.duration)

    print(f"Sending {total_requests} requests at {args.rate} req/s to {args.url}")
    print(f"Duration: {args.duration}s  |  Ctrl-C to stop early\n")

    start = time.perf_counter()
    report_at = start + 5.0

    for i in range(total_requests):
        tick = start + i * interval
        now = time.perf_counter()
        if tick > now:
            time.sleep(tick - now)

        threading.Thread(target=_send, args=(args.url,), daemon=True).start()

        now = time.perf_counter()
        if now >= report_at:
            with _lock:
                total = _counters["total"]
                ok = _counters.get("status_200", 0)
                avg_ms = (_counters["latency_sum_ms"] / total) if total else 0
            elapsed = now - start
            print(f"  {elapsed:5.1f}s | sent={i+1:4d} completed={total:4d} "
                  f"ok={ok:4d} avg_latency={avg_ms:.0f}ms")
            report_at = now + 5.0

    # Wait briefly for in-flight requests
    time.sleep(2)

    elapsed = time.perf_counter() - start
    with _lock:
        total = _counters["total"]
        ok = _counters.get("status_200", 0)
        avg_ms = (_counters["latency_sum_ms"] / total) if total else 0
        statuses = {k: v for k, v in _counters.items() if k.startswith("status_")}

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Completed : {total}")
    print(f"  200 OK    : {ok}")
    print(f"  Avg latency: {avg_ms:.0f} ms")
    print(f"  By status : {statuses}")


if __name__ == "__main__":
    main()
