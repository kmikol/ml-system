# Restrictive Concurrency Limit in Serving Service

**Labels**: `performance`, `serving`, `configuration`

## Description

The serving service uses `asyncio.Semaphore(1)` which limits concurrent requests to exactly 1 at a time. This severely restricts throughput and causes high latency under load, even though the comment suggests the service "saturates at ~3 RPS".

## Affected Files

- `serving/main.py` (line 47)

## Problem Details

Current implementation:

```python
_concurrency = asyncio.Semaphore(1)  # Only 1 concurrent request!
```

This means:
- All requests are serialized
- No parallel processing
- Latency = number_of_requests × per_request_time
- Maximum theoretical RPS is limited to 1 / per_request_time

For example, if each request takes 100ms:
- Max RPS = 1 / 0.1 = 10 RPS (theoretical)
- But with overhead and context switching, actual is likely ~3-5 RPS

## Impact

- **Severity**: Medium (Performance issue)
- **Likelihood**: High (affects all production deployments)
- Can cause:
  - Poor performance under load
  - High latency for concurrent requests
  - Limited throughput
  - Poor resource utilization (CPU/GPU idle)
  - Need for horizontal scaling instead of vertical optimization

## Performance Analysis

With `Semaphore(1)`:
```
Request A: |----100ms----|
Request B:                  |----100ms----|
Request C:                                   |----100ms----|
Total time: 300ms for 3 requests = 10 RPS max
```

With `Semaphore(4)`:
```
Request A: |----100ms----|
Request B: |----100ms----|
Request C: |----100ms----|
Request D: |----100ms----|
Request E:                  |----100ms----|
Total time: 200ms for 5 requests = 25 RPS
```

## Recommended Actions

### 1. Benchmark Current Performance

```python
# Add metrics to measure:
# - Request queue depth
# - Wait time in semaphore
# - Actual inference time
# - CPU/GPU utilization

import time

async def predict_endpoint(request: PredictionRequest):
    queue_start = time.time()
    async with _concurrency:
        queue_time = time.time() - queue_start
        # Log/metric: queue_time

        inference_start = time.time()
        result = await predict(request)
        inference_time = time.time() - inference_start
        # Log/metric: inference_time

        return result
```

### 2. Determine Optimal Concurrency

Test with different semaphore values:
- 1 (current)
- 2
- 4
- 8
- 16

Measure:
- Throughput (RPS)
- Latency (p50, p95, p99)
- CPU/GPU utilization
- Memory usage

### 3. Make Concurrency Configurable

```python
import os

_max_concurrency = int(os.environ.get("SERVING_MAX_CONCURRENCY", "1"))
_concurrency = asyncio.Semaphore(_max_concurrency)
```

Update helm chart:
```yaml
# values.yaml
serving:
  maxConcurrency: 4  # Tune based on benchmarks
```

### 4. Consider Alternative Approaches

Option A: Remove semaphore if thread-safety is fixed
```python
# If ModelManager is properly thread-safe (see issue #XXX)
# and ONNX runtime supports concurrent inference,
# remove the semaphore entirely
```

Option B: Use batching instead of limiting concurrency
```python
# Accumulate requests and batch them
# Process batches of N requests together
# This is more efficient for ML inference
```

Option C: Use connection pooling pattern
```python
# Create a pool of model instances
# Each instance handles 1 request at a time
# But multiple instances allow concurrency
```

## Root Cause

This appears to be a workaround for thread-safety issues in the ModelManager or ONNX runtime. The restrictive limit ensures only one inference runs at a time, avoiding race conditions.

**Related Issues:**
- See issue about ModelManager thread safety (#XXX)
- Should be increased after fixing concurrency bugs

## Testing Recommendations

1. Load test with different concurrency values
2. Measure actual hardware utilization
3. Test for race conditions at higher concurrency
4. Profile to find actual bottlenecks
5. Consider using locust or similar tools for load testing

## Additional Notes

From the code comments, the service "saturates at ~3 RPS". This suggests:
- The real bottleneck is not the semaphore but the inference itself
- However, limiting to 1 concurrent request is still too restrictive
- With proper concurrency, should be able to reach higher RPS
- May need GPU batching or other optimizations

## Priority

This should be addressed after fixing the thread-safety issues in ModelManager and database controllers. The semaphore limit appears to be a safety measure to avoid race conditions.

**Recommended order:**
1. Fix thread-safety bugs (Issues #XXX, #XXX)
2. Add concurrency tests
3. Increase semaphore limit gradually
4. Benchmark and tune
5. Make configurable via environment variable
