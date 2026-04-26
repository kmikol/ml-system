# Architecture

## About This System

This is a project designed to demonstrate production-grade MLOps architecture patterns and best practices. Built as a complete, runnable system (not a toy example), it implements an end-to-end machine learning pipeline with:

**[View on GitHub](https://github.com/kmikol/ml-system)** | **[Open Issues](https://github.com/kmikol/ml-system/issues)**

- **Local-first design** — runs entirely on Kubernetes (k3s) for cost-effective experimentation and learning
- **Cloud-native principles** — architected to deploy to cloud platforms with minimal configuration changes
- **Closed-loop learning** — continuously improves models through event-driven retraining triggered by data quality signals
- **Safety-first deployments** — canary rollouts with automated validation before production promotion

**Current State:** Working system running MNIST classification. See [open issues](https://github.com/kmikol/ml-system/issues) for planned improvements.

---

## Design Intent

The architecture follows three core principles that guide decisions throughout the system:

### 1. **Clear Separation of Concerns**

**Principle:** Online serving, offline retraining, and observability operate independently.

**Why:** Decoupling lets each subsystem scale and evolve without tight coupling. Serving responds to immediate user requests; retraining operates on background signals. Observability infrastructure supports both.

**Related Decision:** [ADR 001 — Event-Driven Retraining](../adr/001-event-driven-retraining.md)

### 2. **Facade-Based Storage Integration**

**Principle:** Application code uses abstraction layers (Data Controller, Model Artifact Controller) rather than coupling directly to storage backends (Postgres, MinIO, MLFlow).

**Why:** Facades decouple services from storage technology choices. A service doesn't know if artifacts are in MinIO or S3. We can swap backends, migrate data, or add caching without touching business logic.

**Related Decision:** [ADR 007 — Model Artifact Controller Abstraction](../adr/007-model-artifact-controller-abstraction.md) and [ADR 004 — MLflow Artifact Storage](../adr/004-mlflow-artifact-storage.md)

### 3. **Safe Model Promotion via Canary Rollouts**

**Principle:** New models are validated in production with real traffic before full promotion. Argo Rollouts gradually shifts traffic from stable to canary models; automated analysis (PSI drift comparison) gates promotion.

**Why:** Canary rollouts reduce deployment risk. A bad model reaches only a fraction of users initially. Automated analysis prevents promotion of models that introduce drift.

**Related Decisions:** [ADR 002 — Canary Rollouts](../adr/002-canary-rollouts.md) and [ADR 006 — KEDA Autoscaling](../adr/006-keda-autoscaling.md)

---

## Runtime Snapshots

The system is intended to be inspected while it runs, not only understood from static diagrams.

![Production load test dashboard showing request volume, error rate, active pods, latency percentiles, PSI, and class-frequency drift](../fig/ml-system-production%20load%20test.png)

**Principle 1 under load.** During a load test, all three subsystems have to cooperate simultaneously — serving handles incoming requests, autoscaling reacts to demand, and observability surfaces signals from both. This view validates that each is operating independently and correctly.

![Canary rollout dashboard showing traffic split, production and canary pod counts, request rate, latency, PSI, and confidence by stage](../fig/ml-system-canary%20rollout.png)

**Principle 3 in action.** The stable and candidate models run side by side, with Argo Rollouts shifting traffic between them while automated drift analysis decides whether to promote. The rollout state is made explicit — traffic share, pod counts, latency, PSI, and confidence — so the promotion decision is observable and auditable rather than implicit.

---

## C4 Model Levels

- **[C4 Level 1: System Context](c4-level-1.md)** — System boundaries, external actors, and high-level information flows
- **[C4 Level 2: Containers](c4-level-2.md)** — Major services, databases, and detailed component interactions organized by operational concern