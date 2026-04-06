# Architecture Decision Records

Architecture Decision Records (ADRs) document significant design choices made in this system, including the rationale and consequences of each decision.

## Overview

Each ADR follows a standard format:
- **Status**: Accepted, Proposed, Superseded, or Deprecated
- **Context**: The problem or situation that required a decision
- **Decision**: What was chosen and why
- **Rationale**: The deeper reasoning behind the choice
- **Consequences**: Positive and negative impacts
- **Alternatives**: Other options that were considered and rejected

## Current ADRs

| ID | Title | Status | Summary |
|----|-------|--------|---------|
| [001](001-event-driven-retraining.md) | Event-Driven Retraining vs Scheduled Cron | Accepted | Retraining triggers when specific conditions are met (drift + labels), not on a schedule |
| [002](002-canary-rollouts.md) | Canary Rollouts vs Blue-Green Deployment | Accepted | New models gradually shift traffic before full promotion, enabling early regression detection |
| [003](003-local-kubernetes.md) | Local Kubernetes (k3d) vs Docker Compose | Accepted | Development environment uses k3d for environment parity with production Kubernetes |
| [004](004-mlflow-artifact-storage.md) | MLflow for Model Artifact Management | Accepted | Centralized model versioning with PostgreSQL metadata store and S3/MinIO artifact backend |
| [005](005-prometheus-monitoring-alerting.md) | Prometheus-Based Monitoring & Alerting | Accepted | Custom metrics with Prometheus + AlertManager + Argo Events for multi-condition orchestration |
| [006](006-keda-autoscaling.md) | KEDA Autoscaling with Arrival Rate Metric | Accepted | Request arrival rate (not CPU) drives autoscaling for concurrency-limited workloads |
| [007](007-model-artifact-controller-abstraction.md) | ModelArtifactController Facade Boundary | Proposed | Standardizes artifact/registry access behind a protocol + facade, with MLflow as current backend |
| [008](008-onnx-model-export.md) | ONNX Model Export vs PyTorch Native | Accepted | Models exported to ONNX format for framework-agnostic serving and reduced deployment dependencies |

## When to Reference These

- **Before modifying alert thresholds** (PSI, annotation count): See ADR-001
- **Before changing deployment strategy** (e.g., shadow deployment, instant rollout): See ADR-002
- **Before moving the system to a different infrastructure**: See ADR-003
- **When wondering "why is this designed this way?"**: Check the relevant ADR

## How to Add New ADRs

1. Create a new file: `00N-short-title.md` (increment the number)
2. Use the Full format with all sections
3. Aim for 800-1200 words (moderate depth)
4. Link related ADRs in the "Related Decisions" section
5. Update this index

## Design Philosophy

The system was built with these principles:

1. **Learning first**: Each design choice prioritizes teaching production ML operations patterns
2. **Local but portable**: Use real tools and infrastructure that transfer to production clusters
3. **Event-driven automation**: React to system state, not schedules
4. **Safety by default**: Gradual rollouts, performance gating, and automatic abort conditions
5. **Observable**: Metrics and alerts make the system transparent
