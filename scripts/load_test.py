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
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict

PAYLOAD = json.dumps(
    {
        "features": {
            "age": 35.0,
            "income": 55000.0,
            "credit_score": 720.0,
            "debt_ratio": 1.2,
            "num_accounts": 5.0,
        }
    }
).encode()

HEADERS = {"Content-Type": "application/json"}

_lock = threading.Lock()
_counters: dict[str, int] = defaultdict(int)


def _send(url: str) -> None:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=PAYLOAD, headers=HEADERS, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception:
        status = 0
    elapsed = time.perf_counter() - t0
    with _lock:
        _counters["total"] += 1
        _counters[f"status_{status}"] += 1
        _counters["latency_sum_ms"] += int(elapsed * 1000)


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
    parser.add_argument("--url", default="http://localhost:8000/predict")
    args = parser.parse_args()

    total_expected_final = _total_expected(
        args.duration, args.rate, args.ramp_up, args.duration, args.ramp_down
    )
    print(f"Peak rate   : {args.rate} req/s")
    print(
        f"Duration    : {args.duration}s  "
        f"(ramp-up {args.ramp_up}s / steady / ramp-down {args.ramp_down}s)"
    )
    print(f"Total       : ~{total_expected_final:.0f} requests")
    print(f"Target      : {args.url}")
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
                avg_ms = (_counters["latency_sum_ms"] / total) if total else 0
            rate = _current_rate(elapsed, args.rate, args.ramp_up, args.duration, args.ramp_down)
            print(
                f"  {elapsed:5.1f}s | rate={rate:5.1f} sent={sent:5d} "
                f"completed={total:5d} ok={ok:5d} avg_latency={avg_ms:.0f}ms"
            )
            report_at = now + 5.0

        # Fire if we're behind the expected cumulative count.
        target = _total_expected(elapsed, args.rate, args.ramp_up, args.duration, args.ramp_down)
        if sent < target:
            threading.Thread(target=_send, args=(args.url,), daemon=True).start()
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
        avg_ms = (_counters["latency_sum_ms"] / total) if total else 0
        statuses = {k: v for k, v in _counters.items() if k.startswith("status_")}

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Sent        : {sent}")
    print(f"  Completed   : {total}")
    print(f"  200 OK      : {ok}")
    print(f"  Avg latency : {avg_ms:.0f} ms")
    print(f"  By status   : {statuses}")


if __name__ == "__main__":
    main()
