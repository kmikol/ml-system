# Architecture: Why These Choices

This page explains the "why" behind the system design. Not just what it does, but why it's built this way.

## Core architectural principle: Local development, production-portable design

The system executes on a single development machine while employing production-grade technologies and patterns.

**Implementation details**:

- **Local Kubernetes**: k3d (lightweight k3s distribution) for local cluster execution
- **Full service deployments**: Real PostgreSQL, MLflow, and other services—not simulations
- **Production patterns**: Argo Workflows for orchestration, Argo Rollouts for deployments, not custom scripts
- **Infrastructure as Code**: Helm charts and YAML manifests matching production deployment formats

**Design rationale**: Developers gain understanding of production technologies through hands-on building. When transitioning to real clusters, the same design principles apply with minimal architectural changes.

## Event-driven automation vs scheduled execution

Retraining is triggered by runtime conditions rather than time-based schedules.

**Comparison**:

```
Schedule-based approach (not used):
  Fixed time (e.g., daily 8 AM) executes retraining unconditionally
  Wasteful: may retrain when no drift exists
  Unresponsive: cannot handle urgent changes

Condition-based approach (implemented):
  Retraining triggered when: PSI > {{ .Values.alerts.drift.psiThreshold }} AND annotations >= {{ .Values.alerts.annotation.countThreshold }}
  Efficient: runs only when justified
  Responsive: reacts to actual system state
```

**Event flow**:
1. Prometheus alert rules evaluate at regular intervals (15-second cadence from config)
2. Alert rule fires when threshold is met; Alertmanager receives it
3. Alertmanager routes alert to appropriate webhook endpoint
4. Argo Events EventSource receives webhook; event is queued
5. Argo Events Sensor aggregates dependencies (requires both PSI alert AND annotation count alert)
6. When both conditions present, Sensor triggers retraining Workflow
7. Argo Workflows executes the workflow steps in sequence

**Why Argo Events instead of simpler webhook-based approaches?**
- Handles multi-condition correlation (requires both drift and annotation alerts)
- Provides reliable event delivery and persistence
- Integrates directly with Argo Workflows for workflow submission

## Canary deployment strategy

New models are deployed through gradual traffic migration rather than immediate switchover.

**Progression**:
```
Stage 1 (0-2m)   {{ .Values.rollout.canarySteps[0].setWeight }}% traffic to candidate / remaining to current
Stage 2 (2-4m)   {{ .Values.rollout.canarySteps[2].setWeight }}% traffic to candidate / remaining to current
Stage 3 (4+ min)    100% to candidate (if neither stages aborted)
```

**Advantages**:
- Issues detected on small traffic subset before full deployment
- Anomalies trigger automatic abort before widespread impact
- Provides sufficient observation period for metric collection

**Implementation in this system**:
- Argo Rollouts manages two active ReplicaSets (stable and canary)
- NGINX ingress controller distributes traffic based on weights
- Both model versions serve predictions simultaneously from different endpoints
- Monitoring (ml_exporter) computes separate drift metrics per version

**Alternative approaches not selected**:
- Blue-green: instantaneous 100% switchover carries higher risk of undetected regressions
- Shadow/mirroring: doubles infrastructure requirements without safety guarantees
- Production environments strongly prefer canary for validation of unknown model versions

## Performance validation before deployment

Candidates must demonstrate performance improvement against the current deployed model before promotion.

**Validation logic**:
```
During canary period:
  Compute PSI for candidate model predictions
  Compute PSI for current model predictions

If candidate PSI < current PSI (lower is better):
  Promotion approved → complete traffic migration
Else:
  Promotion rejected → abort rollout, retain current model
```

**Rationale**: 
- Training on new data does not guarantee improvement
- Evaluation metrics computed on historical data may not reflect production performance
- Live production validation provides highest confidence before commitment

**Implementation**: During canary phase, both models produce predictions. Separate metrics (`drift_psi_class_distribution{role="stable"}` vs `{role="canary"}`) are computed. Workflow compares these values for promotion decision.

**Failure scenario**:
```
Candidate evaluation indicates no improvement
  ↓
MLflow 'Canary' alias cleared
  ↓  
Argo Rollouts rollout aborted
  ↓
Current model continues serving
  ↓
Operator must investigate training pipeline and retry manually
```

## Drift detection via Population Stability Index

Model performance deterioration due to input distribution shift is detected through PSI computation.

**PSI definition and thresholds**:
```
PSI = measure of divergence between reference and observed distributions

PSI < 0.10   No significant shift
PSI < {{ .Values.alerts.drift.psiThreshold }}   Minor shift  
PSI >= {{ .Values.alerts.drift.psiThreshold }}  Significant shift (alert threshold in this system)
```

**Interpretation**: PSI compares prediction class frequencies from training to production. If distribution has shifted significantly, model assumptions may be violated.

**Advantages of PSI for drift detection**:
- Computationally efficient (single-pass statistics)
- Observable as a single metric (works in any monitoring system)
- Threshold-based alerting (clear operational semantics)
- Exposes both stable and canary variants separately

**Limitations**:
- Detects label-space shift only (class distribution changes)
- Does not detect feature-space shift (patterns change but class distribution unchanged)
- Requires minimum sample count (30) to produce meaningful signal

**Why minimum 30 samples?** PSI with very small sample counts produces high variance (noise appears as drift). With 30+ samples, signal-to-noise ratio reaches acceptable levels for alerting.

## Multi-condition trigger requirement

Retraining requires both drift detection AND data availability signals before execution.

**Trigger conditions**:
```
Condition 1: PsiThresholdBreached
  Detected when: drift_psi_class_distribution > {{ .Values.alerts.drift.psiThreshold }}
  Indicates: evidence of model performance degradation

Condition 2: AnnotationCountReached
  Detected when: annotation_annotated_count >= {{ .Values.alerts.annotation.countThreshold }}
  Indicates: sufficient new training material available

Both conditions required: retraining initiates
```

**Rationale**: 
- Drift alone: justifies updating model but lacks training data
- Annotations alone: provides data but no evidence need for update
- Combined condition: ensures both motivation and material present, preventing wasteful or unfounded retraining

## Autoscaling based on request volume

Serving deployment capacity adjusts dynamically based on incoming request rate.

**Mechanism**:
```
KEDA ScaledObject monitors: sum(rate(predict_arrivals_total[{{ .Values.autoscaling.rateWindow }}]))
  Metric: request arrivals per second ({{ .Values.autoscaling.rateWindow }} window)
  
If RPS > threshold (example: {{ .Values.autoscaling.targetRPS }} RPS per replica):
  Action: add replica (scale proportional to excess load)
  
If RPS < threshold (with stabilization window):
  Action: remove replica (conservative: one pod per period)
```

**Why measure arrivals rather than completions?** Serving enforces concurrency control (bounded semaphore). If throughput is already capped by concurrency, measuring completed requests underrepresents actual demand. Measuring arrivals (before concurrency gate) reveals true load, enabling autoscaler to size capacity appropriately.

**Implementation in serving/main.py**: Arrival counter is incremented before acquiring the concurrency semaphore. This ensures KEDA observes true request arrival rate even when requests are queued waiting for concurrency permits.

## Kubernetes on local machines

Running Kubernetes locally (k3d) is computationally heavier than Docker Compose but provides learning and portability benefits.

**Benefits**:
- Direct experience with production-standard deployment patterns
- Tooling identical to production (kubectl, Helm, Argo CLI)
- Design transitions directly to cloud clusters with no architectural rework
- Integration with production observability tools (Prometheus, not custom code)

## Configuration management via Helm

All orchestration and service configuration is managed through Helm charts.

**Structure**: `helm/ml-system/` contains chart metadata, value templates for local/production, and Kubernetes manifests that reference these values.

**Advantages**:
- Single source of truth for all configuration
- Environment-aware deployment (local vs production values)
- Parameter changes without modifying base manifests
- Standard deployment tool (matches production practice)

**Key configurable parameters**: autoscaling targets, alert thresholds, resource limits, drift detection windows

## Intentional gaps and future extension points

This system implements a complete feedback loop but is intentionally incomplete in supporting infrastructure. Components below exist in production systems but are out of scope for this learning platform:

- **Feature store**: Direct request processing; no feature engineering layer
- **Data warehouse**: PostgreSQL only; no time-series analysis infrastructure
- **Experiment tracking beyond MLflow**: No framework for hyperparameter studies
- **User dashboards**: Metrics exposed to Grafana but no business KPI layer
- **Production annotation**: Labels are manual or mocked; no contract with real annotation systems

**Design decision**: Core closed-loop automation (serve → monitor → retrain → rollout) is fully implemented. The human annotation step remains manual, making it the visible control point where operators decide whether retraining happens.

## Extensions for production deployment

When transitioning to production, the following additions are typical:

1. **Add feature store** (Feast, Tecton): Centralized feature computation and versioning
2. **Scale data infrastructure** (data warehouse: BigQuery, Snowflake): Support larger datasets and historical analysis
3. **Add experiment framework** (Weights & Biases, SageMaker): Hyperparameter search and comparison
4. **Build operational dashboards**: Business KPI visualization, model performance tracking
5. **Integrate annotation platform**: Contract with labeling service or deploy internal platform
6. **Expand canary analysis**: Add statistical tests beyond PSI (AUC, latency percentiles)
7. **Add model interpretability**: SHAP or LIME for regression analysis
8. **Implement security hardening**: Container scanning, network policies, RBAC enforcement, secret management
