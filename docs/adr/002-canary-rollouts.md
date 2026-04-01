# ADR-002: Canary Rollouts Instead of Blue-Green Deployments

**Date**: 2024-03-29  
**Status**: Accepted  
**Deciders**: ML Platform Team

## Context

When a retrained model passes performance validation, the system must deploy it to production. This model has been tested only on historical data and observed canary traffic; it has not served 100% of production traffic. Two deployment strategies were evaluated:

1. **Blue-green**: Instantly switch all traffic from current (blue) to new (green) model in a single cutover
2. **Canary**: Gradually shift traffic to the new model (20% → 40% → 100%), monitoring each stage for regressions

## Decision

The system implements **canary rollouts** using Argo Rollouts to gradually shift traffic from the stable model to the candidate model over 4 minutes:

- **Minute 0**: 20% traffic to canary, 80% to stable
- **Minute 2**: 40% traffic to canary, 60% to stable
- **Minute 4**: 100% traffic to canary (if all stages pass automatic analysis)

Each stage is monitored by comparing drift metrics (PSI) between stable and canary replicas. If canary's PSI degrades relative to stable, the rollout is automatically aborted and traffic returns fully to the stable model.

## Rationale

**Risk reduction**: Deploying a new model to all users simultaneously carries high risk. Even with thorough offline evaluation, bugs, data corruption, or unexpected interactions can occur. Canary deployment detects regressions on a small subset (20% of traffic) before exposing them to all users.

**Observability window**: Gradual traffic shift provides time to collect real-world metric data from the canary model. PSI is recomputed continuously; if the canary model produces a notably different output distribution than the stable model (even if both are technically "correct"), this is detected within minutes.

**Automatic abort**: Unlike manual promotion, canary rollouts can be automated with analysis rules. If the AnalysisTemplate detects drift_psi_class_distribution{role="canary"} > drift_psi_class_distribution{role="stable"}, Argo Rollouts automatically halts rollout and reverts traffic to stable.

**Production alignment**: Real ML platforms use canary deployments. This design teaches the pattern while remaining feasible in a local environment.

## Consequences

**Positive**:
- Early detection of model regressions on real traffic (not just test data)
- Automatic rollback if metrics degrade, no human intervention needed
- Production-like deployment pattern that transfers to real systems
- Audit trail: each canary stage is logged, can investigate what happened at each traffic percentage

**Negative**:
- More complex infrastructure: requires Argo Rollouts, NGINX ingress weight routing, and separate replica sets
- Longer time-to-full-deployment (4 minutes vs. instant blue-green)
- Requires stable and canary models to load different MLflow aliases (role-based selection)
- If a bug is in the training pipeline (not model output), canary may not catch it (both model versions would behave the same way initially)

**Monitoring overhead**:
- ml_exporter must track metrics per role (stable vs. canary)
- AnalysisTemplate must be configured with sensible thresholds
- Requires enough traffic to compute PSI reliably within each 2-minute window (minimum 30 samples per PSI calculation)

## Alternatives Considered

### Alternative 1: Blue-Green Deployment
Instantly switch all traffic from current to new model in a single cutover.

**Pros**: Instant deployment, simpler infrastructure (no dual replicas, no gradual routing)  
**Cons**: High-risk single point of failure; if new model is broken, all users affected immediately; no early detection window; harder to rollback if something goes wrong

### Alternative 2: Shadow Deployment
Route traffic to a new model (shadow) without sending its predictions to users. Use shadow output only for offline analysis.

**Pros**: Zero customer impact if model is bad; very thorough offline comparison before going live  
**Cons**: Infrastructure overhead (extra replicas producing predictions users don't see); delayed feedback (must wait for analysis); more complex than canary; NGINX configuration is heavy (need traffic shadowing, or use Istio which is overkill for local setup)

### Alternative 3: Both Shadow + Canary
Run shadow deployment first, then canary rollout.

**Pros**: Combines safety of shadow with agility of canary  
**Cons**: Significantly more complex; double infrastructure overhead; adds unnecessary delay to deployment in a small system

## Implementation Details

- **Argo Rollouts**: Manages ReplicaSets (stable and canary) for the serving deployment
- **NGINX ingress**: Routes traffic to stable/canary based on weights controlled by Argo Rollouts
- **Role labels**: Kubernetes pod labels inject `serving-role=stable` or `serving-role=canary` via Downward API into each pod
- **Model loading**: `serving/main.py` loads either Production (stable) or Canary (canary) MLflow alias based on serving-role
- **Metrics per role**: ml_exporter splits drift_psi_class_distribution by role label

## Related Decisions

- **ADR-001**: Event-driven retraining determines WHEN deployment happens; canary rollouts determine HOW
- **ADR-005**: Performance gating validates the model is better BEFORE canary rollout starts

## Future Improvements

1. **Shadow deployment**: Add shadow traffic routing via Istio if deeper pre-production analysis becomes necessary
2. **Analysis thresholds tuning**: Monitor canary aborts; adjust PSI threshold if too many false positives or false negatives
3. **Metrics diversity**: Currently uses PSI only; could add latency, error rate, or domain-specific metrics to analysis
4. **Gradual traffic percentages**: Currently fixed at 20%/40%/100%; could make configurable per deployment
5. **Rollback hooks**: Add custom webhooks to notify ops teams during rollouts (currently silent except for metrics changes)
