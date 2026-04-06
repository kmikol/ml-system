# C4 Level 1: System Context

This diagram shows the system boundary, external actors, and high-level information flows.

## System Diagram

<div class="arch-diagram arch-diagram-level1">
	<img class="arch-diagram-light" src="../../fig/ml-system-simplified.light.drawio.svg" alt="ML System Architecture">
	<img class="arch-diagram-dark" src="../../fig/ml-system-simplified.dark.drawio.svg" alt="ML System Architecture">
</div>

## System Description

**Online Prediction Path:**
Clients send prediction requests to the system. The system handles synchronous inference and returns predictions with confidence scores and model metadata. Prediction data is persisted for later analysis.

**Closed-Loop Adaptation:**
In parallel, the system monitors prediction quality and collects operational signals. When data quality issues (drift, distribution shift) are detected, the system triggers the retraining pipeline to generate improved models. Accepted models are safely promoted to production.

**Key Characteristics:**
- **Synchronous serving** — handles immediate prediction requests
- **Offline retraining** — continuously improves models based on feedback and drift signals
- **Observability** — monitors both operational health and model quality
- **Safe promotion** — validates and gradually rolls out new models in production

For detailed component breakdown and interactions, see [C4 Level 2: Containers](c4-level-2.md).
