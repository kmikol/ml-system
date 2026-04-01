# ADR-006: KEDA-Based Autoscaling Using Request Arrival Rate Instead of CPU

**Date**: 2024-03-29  
**Status**: Accepted  
**Deciders**: ML Platform Team

## Context

The serving application must scale automatically when traffic increases and scale down when traffic decreases. Kubernetes provides a native Horizontal Pod Autoscaler (HPA) that scales based on CPU and memory utilization, but this creates problems in the ml-system:

1. **Concurrency-limited workloads**: The serving application enforces a semaphore that limits concurrent inference requests to 1 (with simulated 333ms latency per request). This design simulates realistic ML inference bottlenecks, causing:
   - At low request rates: CPU is idle (waiting for requests to arrive)
   - At high request rates: CPU is high (running inference) **and** requests queue up behind the semaphore
   - CPU utilization alone cannot distinguish between "waiting for input" and "queue is growing"

2. **Simulated vs. Real Load**: The local development environment simulates latency artificially. In real ML systems, the same problem exists: GPU can saturate while CPU remains underutilized during small-batch inference, making CPU an unreliable scaling metric.

3. **Desired Metric**: The true signal for scaling is **request arrival rate** (requests/second), independent of how busy the system is processing them.

## Decision

The system uses **KEDA (Kubernetes Event autoscaling Deployment)** to scale based on a custom Prometheus metric representing request arrival rate:

**Scaling Signal**: `predict_arrivals_total` counter (Prometheus metric)  
**Scaling Query**: `sum(rate(predict_arrivals_total[30s]))`  
**Target**: 5 requests/second per replica  
**Formula**: `desiredReplicas = ceil(totalRPS / 5)`

**Scaling Policies**:
- **Min replicas**: 1 (always at least 1 pod running)
- **Max replicas**: 15 (safety cap)
- **Scale-up**: Immediate, double replicas every 15 seconds (1→2→4→8→...)
- **Scale-down**: Conservative, remove 1 pod every 15 seconds after 60-second stabilization window

```
Serving Pod exports /metrics
  ↓ (every 15s scrape)
Prometheus collects predict_arrivals_total
  ↓ (every 10s poll)
KEDA controller queries Prometheus
  ↓ computes: ceil(rate / 5)
Argo Rollout replicas updated
```

## Rationale

**Request arrival rate reflects true incoming load**: Unlike CPU utilization, the `predict_arrivals_total` metric is incremented **before** the semaphore acquire. It captures the true arrival rate of requests regardless of queue depth or processing state. Two pods might have different CPU utilization (one idle, one saturated) but the same arrival rate metric tells the true story: "we're receiving X requests per second."

**Realistic simulation of ML constraints**: The semaphore (concurrency limit = 1, latency = 333ms) simulates realistic ML inference where throughput is limited by the model's computational cost, not by the server's CPU capacity. This mirrors real scenarios (GPU-bound inference, batch limits) where CPU is not the bottleneck. By using arrival rate instead of CPU, KEDA learns to handle this correctly.

**Decoupling from resource utilization**: Specifying "this deployment should handle 5 RPS per replica" is a clear operational intent, independent of the pod's CPU requests/limits or the node's total capacity. This makes the system portable across different cluster configurations (large nodes, small nodes, different CPU allocations).

**KEDA advantages over native HPA**:
- **Flexible metric sources**: Prometheus allows any query, not just CPU/memory
- **Declarative queries**: Business logic ("5 RPS per pod") is explicit in the Prometheus query, not hidden in HPA thresholds
- **Direct field integration**: KEDA integrates with Prometheus directly; future metrics (latency percentiles, queue depth, etc.) can be added without infrastructure changes
- **Immediate feedback**: HPA often requires CPU to reach threshold before scaling; KEDA can scale preemptively based on metric trends

**Aggressive scale-up, conservative scale-down**:
- Scale-up is immediate (no stabilization window) with 100% growth per 15s: prioritizes latency (better to have spare capacity than to queue requests)
- Scale-down is conservative (60s stabilization window, 1 pod per 15s): prevents thrashing if traffic fluctuates (e.g., bursty workload)

**30-second rate window**: The Prometheus query `rate(predict_arrivals_total[30s])` uses a 30-second window (2x the 15-second scrape interval). This smooths transient spikes and prevents KEDA from oscillating due to momentary traffic bursts.

## Consequences

**Positive**:
- Fast scale-up response (exponential growth: reaches 15x replicas in 45 seconds if needed)
- Responsive to actual workload (arrival rate metric captures true load)
- Decoupled from pod resource requests (works for CPU-bound, IO-bound, GPU-bound workloads)
- Extensible: can add multiple KEDA triggers (e.g., also scale on p99 latency)
- Predictable behavior: "5 RPS per pod" is clear and easy to monitor

**Negative**:
- Additional infrastructure: KEDA controller pod must be running on the cluster
- Prometheus dependency: if Prometheus is down, KEDA falls back to previous desired state (suspected, not fully tested)
- Metric-dependent: scaling is broken if metric collection stops (Alloy scraping fails, Prometheus becomes unreachable)
- Scale-down delay: Conservative policy means spare capacity may persist for up to 75 seconds (60s stabilization + max 15s before first evaluation)
- Tuning required: The 5 RPS target and scale policies were heuristically selected, not tuned based on production traffic patterns

**Operational requirements**:
- Monitor Prometheus uptime; KEDA becomes ineffective if metrics stop flowing
- Verify KEDA controller pod is running and healthy
- Monitor actual request arrival rates and adjust `targetRPS: 5` if workload profile changes
- Review scale-up/down policies quarterly, especially if traffic patterns shift

## Alternatives Considered

### Alternative 1: Native Kubernetes HPA with CPU Utilization
Use built-in HPA set to scale when CPU exceeds 70%.

**Pros**: 
- Built into Kubernetes, no additional controllers
- Works for CPU-bound workloads

**Cons**:
- CPU is not a good signal for concurrency-limited or GPU-bound workloads
- In the simulated environment: won't scale properly once semaphore is active
- In real ML systems: GPU can saturate while CPU remains low
- Scale-up is often too late (only after CPU threshold is breached)

### Alternative 2: Manual Replica Adjustment
Operations team manually adjusts replicas based on monitoring dashboard.

**Pros**: Full visibility and control

**Cons**: 
- Doesn't scale (requires 24/7 monitoring)
- Slow response to traffic changes
- Prone to human error
- Not production-like

### Alternative 3: Request Queue Depth Metric
Scale based on how many requests are queued in the semaphore rather than arrival rate.

**Pros**: Directly reflects system saturation

**Cons**:
- Lagging indicator (queue only grows after pods are saturated)
- Scale-up happens too late
- Opposite approach: would be better as a "backpressure" signal than a primary scaling metric

## Implementation Details

**Metric Export in Serving** ([serving/main.py](serving/main.py)):

```python
from prometheus_client import Counter

# Incremented BEFORE semaphore to capture true arrival rate
_predict_arrivals = Counter(
    "predict_arrivals_total",
    "Predict requests at arrival",
    labelnames=["serving_role"]  # stable or canary
)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    _predict_arrivals.labels(serving_role=SERVING_ROLE).inc()  # ← Incremented first
    
    # Simulate ML inference latency with concurrency limit
    async with _concurrency:  # Semaphore(1), capped at ~3 RPS
        # ... inference logic ...
```

**KEDA ScaledObject** ([helm/ml-system/templates/scaledobject.yaml](helm/ml-system/templates/scaledobject.yaml)):

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: fastapi-serving-scaler
  namespace: ml-system
spec:
  scaleTargetRef:
    apiVersion: argoproj.io/v1alpha1
    kind: Rollout
    name: fastapi-serving
  minReplicaCount: 1
  maxReplicaCount: 15
  pollingInterval: 10              # Query Prometheus every 10s
  cooldownPeriod: 30
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.ml-system.svc.cluster.local:9090
        metricName: predict_arrivals_per_second
        query: sum(rate(predict_arrivals_total[30s]))
        threshold: "5"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100               # Double current replicas
          periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 1                 # Remove 1 pod
          periodSeconds: 15
```

**Autoscaling Configuration** ([helm/ml-system/values.yaml](helm/ml-system/values.yaml)):

```yaml
autoscaling:
  enabled: true
  minReplicas: 1                   # Always at least 1 pod
  maxReplicas: 15
  targetRPS: 5                     # Replicas = ceil(totalRPS / 5)
  rateWindow: "30s"
```

**Metric Collection Pipeline**:
1. Serving pods export `/metrics` (port 8000)
2. Alloy scrapes every 15 seconds (discovers pods by label: `app=fastapi-serving`)
3. Prometheus receives metrics via remote-write from Alloy
4. KEDA polls Prometheus every 10 seconds
5. KEDA computes desired replicas: `ceil(sum(rate(...)) / 5)`
6. KEDA patches Argo Rollout `spec.replicas`

## Related Decisions

- **ADR-005**: Prometheus metrics collection provides the `predict_arrivals_total` metric
- **ADR-002**: Argo Rollouts primitive enables KEDA to control replicas via scaleTargetRef

## Future Considerations

1. **Production tuning**: Measure actual RPS distribution and adjust `targetRPS: 5` based on real workload
2. **Multi-metric scaling**: Add latency percentile (p99) as secondary trigger ("scale if RPS > 5 OR p99 > 500ms")
3. **Scale-to-zero**: Set `minReplicaCount: 0` for cost optimization during idle periods
4. **Predictive scaling**: Use historical traffic patterns to pre-scale before peak hours
5. **Failure recovery**: Test and document Prometheus outage behavior; consider fallback mechanism
6. **Queue depth metric**: Export semaphore queue length; use as observability signal (not scaling)
