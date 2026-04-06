# ADR-001: Event-Driven Retraining Instead of Scheduled Cron Jobs

**Date**: March 2026
**Status**: Accepted  
**Deciders**: kmikol

## Context

The system requires an automated mechanism to trigger model retraining when the deployed model begins to degrade due to data distribution shifts. Three approaches were evaluated:

1. **Manual trigger**: ML engineers manually decide when to retrain and submit jobs via CLI or API.
2. **Scheduled (Cron)**: Retraining runs on a fixed schedule (e.g., daily, weekly) regardless of system state.
3. **Event-driven**: Retraining is triggered automatically when specific conditions are met (drift detected + sufficient labeled data available).

## Decision

The system implements **event-driven retraining** using Argo Events to correlate alerts and submit workflows automatically when both conditions fire:
- Drift detected: Population Stability Index (PSI) > 0.25 on recent predictions
- Annotation available: 50 or more newly labeled predictions in the dataset

When both alerts fire within a recent window, a retraining workflow is automatically submitted to Argo Workflows.

## Rationale

**Learning alignment**: Event-driven architectures reflect how production ML systems actually work. Building this teaches transportable skills applicable to real platforms.

**Operational efficiency**: Unlike scheduled retraining, this approach only triggers when both conditions are genuinely met:
- Drift alone without labels → no retraining (prevents wasteful compute)
- Labels available without drift → no retraining (current model is performing)
- Both conditions → retraining proceeds (makes business sense)

**Condition encoding**: The system encodes ML engineering judgment into metric thresholds rather than manual decision-making. PSI > 0.25 is an industry-accepted threshold for significant distribution shift (vs. minor noise).

**Argo Events choice**: Direct Prometheus webhooks could trigger workflows, but Argo Events adds essential capabilities:
- **Multi-condition correlation**: Sensors wait for multiple alert dependencies before triggering
- **Event replay**: If the webhook service temporarily fails, Argo Events preserves event history
- **Decoupling**: Prometheus rules don't need to know about Argo infrastructure; Argo Events handles integration

## Consequences

**Positive**:
- Compute resource usage correlates with system need (no idle retraining runs)
- Faster response to real degradation vs. daily/weekly schedules
- Clear audit trail: alert rules define exactly when retraining occurs
- Enables manual override: engineers can still submit retrain workflows manually when needed

**Negative**:
- Added complexity: Argo Events configuration, alert rule specification, and condition coordination
- Requires explicit encoding of domain knowledge into thresholds (PSI > 0.25)
- Harder to debug if alerts don't fire when expected (need to check Prometheus rules, webhooks, EventSource configuration)
- Potential latency between drift detection and retraining start (minutes-level, not critical)

**Maintenance**:
- PDI threshold (0.25) may need tuning based on observed behavior over time
- If business requirements change (e.g., more aggressiveness on degradation), alert rules must be updated

## Alternatives Considered

### Alternative 1: Manual Trigger Only
ML engineers manually decide when to retrain and submit jobs.

**Pros**: Simple, no automation infrastructure needed, full human control  
**Cons**: Doesn't scale (requires human vigilance 24/7), slow response time, can't be automated in alerting on weekends/nights, not production-like

### Alternative 2: Scheduled Cron Jobs
Retraining runs daily or weekly on a fixed schedule.

**Pros**: Simple to implement (just a Cron rule and a script), predictable execution time  
**Cons**: Wasteful compute when no drift occurs; misses urgent degradations between scheduled runs; doesn't learn event-driven patterns

## Related Decisions

- **ADR-002**: Choice of canary rollouts provides the safety gate for event-driven retraining (candidate validation before production promotion)
- **ADR-003**: Dual-condition trigger (this ADR) + PSI-based drift detection refines when retraining fires

## Future Considerations

1. **Additional trigger conditions**: Extend beyond PSI to include feature-space drift, latency degradation, or other metrics
2. **Threshold tuning**: Monitor retraining frequency; adjust PSI threshold if patterns suggest it's too aggressive or too lenient
3. **Manual override prioritization**: Consider queuing/priority system if multiple workflows can trigger simultaneously
4. **Failure handling**: Decide behavior if a retrain workflow fails (should a second one retry automatically, or require manual intervention?)
