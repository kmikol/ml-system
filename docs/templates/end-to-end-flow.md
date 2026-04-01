# System Operation

This page describes the operational flow of the system from initial deployment through model improvement cycles.

## Model deployment and online serving

Assume a trained model is deployed to the serving environment.

### Phase 1: Normal operation (serving + monitoring)

```
[Client] → request prediction
  ↓
[Serving] → loads v1 model from MLflow
  ↓ (makes prediction, adds latency metric)
  ↓
[Logging] → writes prediction to PostgreSQL
  ↓
[Prometheus] scrapes serving metrics (p99 latency, prediction class distribution)
  ↓
[Monitoring] polls PostgreSQL every 5 seconds
  ↓ (computes PSI on recent predictions)
  ↓
[Metrics] exposed as drift_psi_class_distribution gauge
```

Continuous operation state:
- Request traffic arriving at serving endpoints
- Prediction responses returned to clients
- Prediction events logged to persistent storage
- Prometheus scrapes metrics from serving  
- Monitoring service polls storage every 5 seconds
- Drift metric (PSI) computed and exposed

**Observable metrics**:
- `predict_arrivals_total`: cumulative request counter
- `drift_psi_class_distribution`: current PSI value, expected < 0.1
- `http_request_duration_seconds`: latency distribution, p99 typically < 500ms

### Phase 2: Drift detection

Over extended operation, real-world data distribution may shift relative to training distribution. This is detected through PSI computation.

When PSI exceeds configured threshold (typically {{ .Values.alerts.drift.psiThreshold }}):

```
[Prometheus] evaluates alert rule → PsiThresholdBreached condition satisfied
[Alertmanager] routes alert  
[Argo Events EventSource] receives webhook notification
[Event] queued in event system
```

Alert fires but retraining does not yet commence. Fresh training data (labels) is required to avoid retraining on insufficient evidence.

### Phase 3: Annotation availability

In parallel with monitoring, predictions are being labeled (through human annotation, labeling service, or other means) and annotations are recorded in storage.

When the count of labeled predictions reaches the configured threshold ({{ .Values.alerts.annotation.countThreshold }}):

```
[Storage] records accumulated labels
[Monitoring] detects annotation count >= threshold  
[Prometheus] updates annotation_annotated_count metric
[Alert rule] AnnotationCountReached fires
[Argo Events EventSource] receives webhook
[Event] queued in event system
```

### Phase 4: Retraining trigger condition

The Argo Events Sensor aggregates both alert events. When both are present:

```
PsiThresholdBreached (PSI > {{ .Values.alerts.drift.psiThreshold }}) present
AnnotationCountReached (>={{ .Values.alerts.annotation.countThreshold }} labeled predictions) present
  ↓
[Argo Events Sensor] aggregates dependencies
  ↓
[Workflow] retraining WorkflowTemplate submitted to Argo Workflows
```

This represents event-driven automation: retraining initiates only when explicit conditions are satisfied. No manual intervention required.

### Phase 5: The retraining workflow (6 sequential steps)

The retrain workflow now runs:

**Step 1: Integrate annotations**
```
Read: all labeled predictions from DB
Write: build a new dataset version with new labels merged in
```

**Step 2: Train**
```
Read: dataset version produced by integration step
Execute: model training on updated data
Output: trained model registered in MLflow as new run
```

**Step 3: Tag candidate version**
```
Action: assign 'Canary' alias to newly trained model in MLflow
Effect: Serving pods with canary role will load this model version
```

**Step 4: Initiate gradual rollout**
```
Action: trigger Argo Rollouts to begin canary deployment
Effect: new canary ReplicaSet created with candidate model; traffic begins gradual migration
```

**Step 5: Monitor canary period**
```
Duration: canary deployment in progress (typically 4+ minutes based on step configuration)
Metrics collected: PSI and performance metrics for both stable and canary models
```

**Step 6: Promotion decision**
```
Evaluation: compare candidate PSI against currently-deployed model PSI

If candidate performs better (PSI improved):
  → Tag as 'Production' alias (promotion)
  → Argo Rollouts completes traffic migration to 100% canary
  
If candidate does not improve (PSI same or worse):
  → Clear 'Canary' alias (rejection)
  → Argo Rollouts aborts; traffic remains on stable model
```

### Phase 6: Promotion complete

Following successful promotion, the serving deployment transitions to the candidate model.

```
[Client] → requests come in
  ↓
[Serving] → loads v2 from MLflow (Production alias)
  ↓
[Monitoring] → tracks new PSI for v2 model
  ↓
[Cycle repeats if data drifts again...]
```

### Phase 6b: Rollback and retry

If the candidate model does not outperform the current model, promotion is rejected.

```
[Workflow promotion step] evaluates candidate vs current PSI
[Decision] PSI did not improve
  ↓
[MLflow] 'Canary' alias cleared
  ↓
[Argo Rollouts] rollout aborted; traffic restored to stable model
  ↓
[Serving] continues with existing model version
```

Retrying requires manual intervention: debug training logic and re-submit workflow manually.

## Canary rollout progression

During the candidate evaluation period, Argo Rollouts gradually shifts traffic according to configured steps:

Traffic distribution during canary phase:
```
Initial (0:00)   → {{ .Values.rollout.canarySteps[0].setWeight }}% canary / remaining stable ({{ .Values.rollout.canarySteps[1].pause.duration }} hold)
Phase 2 ({{ .Values.rollout.canarySteps[1].pause.duration }})   → {{ .Values.rollout.canarySteps[2].setWeight }}% canary / remaining stable ({{ .Values.rollout.canarySteps[3].pause.duration }} hold)  
Phase 3 (Total)  → 100% canary (if evaluation succeeds)
```

Throughout progression, separate PSI metrics are computed for each variant:
- `drift_psi_class_distribution{role="stable"}`: PSI of stable model predictions
- `drift_psi_class_distribution{role="canary"}`: PSI of canary model predictions

If canary metrics indicate regression, Argo Rollouts can abort the rollout before completing the final traffic shift.

## Independent autoscaling

Serving deployment autoscaling operates independently of model updates. KEDA continuously monitors request arrival rate:

```
Monitoring interval: 10 seconds
  Query: sum(rate(predict_arrivals_total[{{ .Values.autoscaling.rateWindow }}]))
  Value: requests per second over last {{ .Values.autoscaling.rateWindow }}
  
  If RPS > threshold (configured example: {{ .Values.autoscaling.targetRPS }} RPS per replica):
    Scale up: add replica (percent-based increase)
  
  If RPS < threshold with stabilization window satisfied:
    Scale down: remove replica (one per interval)
```

Autoscaling continues independent of retraining operations.

## Manual retraining

Retraining can be initiated manually without waiting for automatic trigger conditions:

```bash
# Submit retraining workflow directly
argo submit k8s/argo/workflows/retrain.yaml  -n argo
```

Manual submission bypasses automatic trigger conditions but applies the same validation: candidate model must outperform current production model before promotion.

## What alerts mean

| Alert | Means | You should |
|-------|-------|-----------|
| `HighLatency` | p99 latency > {{ .Values.alerts.latency.p99ThresholdSeconds }} second for {{ .Values.alerts.latency.forDuration }} | Check serving logs, increase replicas, profile model |
| `PsiThresholdBreached` | PSI > {{ .Values.alerts.drift.psiThreshold }} (distribution shift) | Expect retraining to trigger once labels are ready |
| `AnnotationCountReached` | {{ .Values.alerts.annotation.countThreshold }}+ pending predictions labeled | Expect retraining to trigger once drift is detected |

If either of the last two fires alone, nothing happens yet. You need both before retraining kicks off.

## Where the actual code lives

- **Serving predictions**: `serving/main.py` (FastAPI server, metrics emission)
- **Drift calculation**: `monitoring/ml_exporter/main.py` (PSI computation, 30-sample minimum window)
- **Retraining steps**: `k8s/argo/workflows/retrain.yaml` (integrated scripts in each step)
- **Alert definitions**: `helm/ml-system/templates/prometheus.yaml` (alert rules)
- **Alert routing**: `helm/ml-system/templates/alertmanager.yaml` (which alert goes where)
- **Event automation**: `k8s/argo/argo-events-resources.yaml` (webhooks → Argo Events → Sensor → Workflow)
- **Autoscaling config**: `helm/ml-system/templates/scaledobject.yaml` (KEDA ScaledObject for RPS-based scaling)
