# ADR-005: Prometheus + AlertManager + Argo Events for Multi-Alert Orchestration

**Date**: March 2026
**Status**: Accepted  
**Deciders**: kmikol

## Context

The system must monitor deployed models and automatically trigger retraining when specific conditions are met. This requires:

1. **Drift detection**: Continuous monitoring of prediction distribution shifts
2. **Data collection**: Tracking when sufficient labeled data is available for retraining
3. **Multi-condition coordination**: Retraining should trigger only when BOTH drift is detected AND sufficient labels exist (not just one)
4. **Production alignment**: Monitoring patterns should transfer to production systems

Three architectural approaches were evaluated:

1. **Custom polling**: Serving application directly polls database and calls workflow webhook
2. **Prometheus direct webhooks**: Prometheus alert rules send webhooks directly to Argo Workflows API
3. **Prometheus + AlertManager + Argo Events**: Multi-layer architecture with event correlation and replay

## Decision

The system implements **Prometheus + AlertManager + Argo Events** pipeline:

```
Serving Pod (every request)
  ↓ exports Prometheus metrics
/metrics endpoint (Prometheus format)
  ↓ (scraped by Alloy every 15s)
Alloy Agent
  ↓ remote writes to
Prometheus TSDB (stores 5Gi of time-series)
  ↓ (rules evaluated every 15s)
Alert Rules:
  - HighLatency: p99 > 1.0s for 5m
  - PsiThresholdBreached: PSI > 0.25 for 30s
  - AnnotationCountReached: labeled_count >= 50 (instant)
  ↓ (fires alerts to)
AlertManager (port 9093)
  ├─ PsiThresholdBreached → webhook POST to psi-alert-eventsource:12000/psi-alert
  └─ AnnotationCountReached → webhook POST to annotation-ready-eventsource:12001/annotation-ready
  ↓
Argo Events EventSource (listens for webhooks)
  ├─ psi-alert-eventsource → triggers sample-and-label workflow
  └─ annotation-ready-eventsource → stored in NATS EventBus
  ↓
Argo Events Sensor (conditions: psi-alert && annotation-ready)
  ├─ Only triggers retrain workflow when BOTH events present
  └─ Submits to Argo Workflows API
  ↓
Argo Workflows (in argo namespace)
  ├─ sample-and-label (runs when drift detected)
  └─ retrain (runs only when both conditions met)
```

**Three alert rules**:

| Alert | Metric | Condition | For | Action |
|-------|--------|-----------|-----|--------|
| **HighLatency** | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))` | > 1.0s | 5m | Grafana annotation only |
| **PsiThresholdBreached** | `drift_psi_class_distribution{role="stable"}` | > 0.25 | 30s | Webhook to psi-alert-eventsource |
| **AnnotationCountReached** | `annotation_annotated_count` | >= 50 | immediate | Webhook to annotation-ready-eventsource |

## Rationale

**Prometheus for observability**: Prometheus is the Kubernetes ecosystem standard for metrics collection. It provides:
- **Pull-based scraping**: Serving exports `/metrics` endpoint; Prometheus scrapes the data. No instrumentation needed in serving code beyond `prometheus_client` Counter/Gauge.
- **Transparent metric history**: Time-series database retains 5 days of metrics for debugging ("why did the alert alert fire at 3pm?")
- **Flexible queries**: PromQL allows complex aggregations (`sum(rate(...))`, `histogram_quantile()`) without rewriting application code

**AlertManager for alert deduplication and routing**: AlertManager sits between Prometheus and event systems to:
- **Group instances**: Multiple pods firing the same alert becomes one notification (reduces webhook storm)
- **Repeat interval**: Re-sends webhook every 3 minutes while alert is active (webhooks are unreliable; repetition ensures delivery)
- **Routing logic**: Different alerts route to different EventSources without hard-coding URLs in Prometheus

**Argo Events for multi-condition correlation**: The key innovation is requiring BOTH alerts to fire before triggering retraining:

```yaml
# Sensor definition (Argo Events)
spec:
  dependencies:
    - name: psi-alert-dep
      eventSourceName: psi-alert
      eventName: psi-alert
    - name: annotation-ready-dep
      eventSourceName: annotation-ready
      eventName: annotation-ready
  
  triggers:
    - template:
        name: retrain-trigger
        argoWorkflow:
          operation: submit
  
  dependencyGrouping: AND  # BOTH must fire
```

This AND logic cannot be expressed in:
- Direct Prometheus webhooks (would need custom application logic to receive and correlate two webhooks)
- Custom polling code (adds complexity and testing burden)

Argo Events achieves this with declarative configuration, no code changes.

**Event replay for resilience**: Argo Events uses NATS as the backing event store. If the Sensor or EventSource pod crashes:
- Events already received are persisted in NATS (default: 1-hour retention)
- On pod restart, Events are re-delivered
- This prevents loss of drift/annotation alerts during transient pod failures

**Metric design choices**:

- **PSI (Population Stability Index) threshold (0.25)**: Arbitrary but industry-standard starting point:
  - PSI < 0.10: Negligible shift
  - PSI 0.10-0.25: Moderate shift (monitor)
  - PSI > 0.25: Significant shift (alert)
  - This is tunable; threshold can be adjusted without code changes (Helm value)

- **Annotation count threshold (50)**: Sized for MNIST (10 classes, 5 samples per class minimum). Also arbitrary starting point tunable via Helm.

- **Latency threshold (1.0s p99)**: Represents SLA target for serving (p99 latency under 1 second means 99% of requests complete in 1s). Helps identify model performance degradation or infrastructure saturation.

**Role-based metrics (stable vs. canary)**: The `drift_psi_class_distribution` metric has a label `role` (stable or canary). This enables:
- During canary rollout: compare new model's distribution against baseline (should be similar)
- ML Exporter and Prometheus queries reference `{role="stable"}` specifically
- If canary PSI suddenly spikes > 0.25, alert fires (indicating candidate model is diverging unexpectedly)

## Consequences

**Positive**:
- **Multi-condition safety**: Retrain only when both conditions met; prevents wasteful compute when either alone is insufficient
- **Event replay resilience**: Transient pod crashes don't lose drift/annotation events
- **Transparent history**: Prometheus stores 5 days of metrics; researchers can replay events or debug "why did it alert?"
- **Ecosystem alignment**: Prometheus + Kubernetes is industry standard; patterns transfer to production systems
- **Decoupled concerns**: Prometheus doesn't embed Argo logic; Argo Events don't embed Prometheus knowledge
- **Low latency detection**: PSI computed every 5 seconds, alerts fire within 30-90 seconds of condition met

**Negative**:
- **Operational complexity**: Multiple moving parts (Prometheus, AlertManager, EventSource, Sensor, NATS EventBus)
- **Prometheus dependency**: If Prometheus scraping stops or Prometheus pod crashes, all alerts fail
- **Eventual consistency**: ~15-30 second latency from metric change to alert fire (depends on scrape interval, evaluation interval, EventSource processing)
- **Storage overhead**: Storing all metrics for 5 days consumes significant disk (5Gi PVC in local setup)
- **Debugging complexity**: Tracing a missed alert requires checking: Prometheus rules, AlertManager router, EventSource webhook HTTP logs, Sensor conditions, NATS EventBus
- **State loss on restart**: If multiple components restart simultaneously, event correlation may be lost (mitigated by NATS replay, but not foolproof)

**Operational requirements**:
- Monitor Prometheus health (if scraping fails, entire monitoring pipeline breaks)
- Monitor AlertManager webhook delivery (test endpoint connectivity to EventSources)
- Monitor NATS EventBus memory (events are stored in-memory; configurable retention)
- Regularly verify that sample-and-label and retrain workflows are submitted as expected

## Alternatives Considered

### Alternative 1: Custom Polling Code
Serving application periodically checks database for annotation count and PSI; calls workflow webhook directly.

**Pros**:
- Simple, no external monitoring infrastructure
- Correlation logic embedded in application code (clear flow)

**Cons**:
- Couples serving application to ML orchestration (serving code must know about Argo)
- No event history or replay
- Doesn't scale if multiple models/services need orchestration
- Not production-like; teaches non-transferable patterns
- Debugging requires reading application logs, not Prometheus

### Alternative 2: Direct Prometheus → Argo Webhooks
Prometheus alert rules send webhooks directly to Argo Workflows API (no AlertManager/Argo Events intermediary).

**Pros**:
- Fewer hops, lower latency
- Simpler architecture (3 components instead of 5)

**Cons**:
- Cannot express multi-condition AND logic without custom code
- Webhook delivery unreliable; no automatic retry
- No deduplication if multiple pods fire same alert
- Argo Workflows API details embedded in Prometheus (tight coupling)
- Not reusable for other workflows/tools

### Alternative 3: Cloud-Managed Observability (Datadog, CloudWatch)
Use cloud provider's monitoring service.

**Pros**:
- Fully managed, no infrastructure needed

**Cons**:
- Adds cloud provider lock-in
- Data leaves cluster (privacy/compliance concerns)
- Costs scale with data volume
- Doesn't teach Kubernetes-native patterns
- Misaligned with local development goal

## Implementation Details

**Metrics Exported by Serving** ([serving/main.py]()):

```python
from prometheus_client import Counter, Gauge, Histogram

# Counters (monotonic, scraped for rate())
predict_arrivals = Counter(
    "predict_arrivals_total",
    "Total predictions at request arrival",
    labelnames=["serving_role"]  # stable or canary
)

# Gauges (current value)
annotation_count = Gauge(
    "annotation_annotated_count",
    "Count of labeled predictions ready for retraining"
)

drift_psi = Gauge(
    "drift_psi_class_distribution",
    "PSI of current prediction distribution vs. baseline",
    labelnames=["role"]  # stable or canary
)

# Histograms (distributions)
request_latency = Histogram(
    "http_request_duration_seconds",
    "Request duration in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)
```

**Prometheus Alert Rules** ([helm/ml-system/templates/prometheus.yaml]()):

```yaml
groups:
  - name: ml_system_alerts
    interval: 15s
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        annotations:
          summary: "P99 latency exceeds 1.0 second"
          action: "Check serving logs, increase replicas"

      - alert: PsiThresholdBreached
        expr: drift_psi_class_distribution{role="stable"} > {{ .Values.alerts.drift.psiThreshold }}
        for: {{ .Values.alerts.drift.forDuration }}
        annotations:
          summary: "Distribution shift detected"

      - alert: AnnotationCountReached
        expr: annotation_annotated_count >= {{ .Values.alerts.annotation.countThreshold }}
        for: {{ .Values.alerts.annotation.forDuration }}
        annotations:
          summary: "Sufficient labels available"
```

**AlertManager Routing** ([helm/ml-system/templates/alertmanager.yaml]()):

```yaml
route:
  receiver: "default"
  routes:
    - match:
        alertname: PsiThresholdBreached
      receiver: psi-alert
      repeat_interval: 3m

    - match:
        alertname: AnnotationCountReached
      receiver: annotation-ready
      repeat_interval: 3m

receivers:
  - name: psi-alert
    webhook_configs:
      - url: http://psi-alert-eventsource-svc:12000/psi-alert
        send_resolved: false

  - name: annotation-ready
    webhook_configs:
      - url: http://annotation-ready-eventsource-svc:12001/annotation-ready
        send_resolved: false

  - name: default
    # HighLatency alerts go to /dev/null (observational only)
```

**Argo Events Sensor** ([k8s/argo/argo-events-resources.yaml]()):

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Sensor
metadata:
  name: retrain-sensor
  namespace: argo-events
spec:
  eventBusName: default
  
  dependencies:
    - name: psi-alert-dep
      eventSourceName: psi-alert
      eventName: psi-alert

    - name: annotation-ready-dep
      eventSourceName: annotation-ready
      eventName: annotation-ready

  triggers:
    - template:
        name: retrain-trigger
        argoWorkflow:
          operation: submit
          parameters:
            - src:
                dependencyName: psi-alert-dep
                dataKey: body
              dest: spec.arguments.parameters.0.value

  # Critical: AND logic means both must fire
  dependencyGrouping: AND
```

## Related Decisions

- **ADR-001**: Event-driven retraining relies entirely on this alert pipeline
- **ADR-006**: KEDA autoscaling uses `predict_arrivals_total` metric exported by this monitoring layer

## Future Considerations

1. **Alert tuning**: Monitor actual PSI distributions and adjust 0.25 threshold based on false positive rate
2. **Feature-space drift**: Add detection beyond class distribution (e.g., embedding-space KL divergence)
3. **Model confidence degradation**: Alert when mean prediction confidence drops below threshold
4. **SLO-based alerting**: Define SLOs ("p99 < 500ms") and alert on SLO violations rather than absolute thresholds
5. **Long-term storage**: Archive metrics to S3 after 5-day retention for quarterly analysis
6. **Distributed tracing**: Add OpenTelemetry for request-level debugging across services
