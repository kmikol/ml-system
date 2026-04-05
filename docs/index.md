# MLOps System

An end-to-end MLOps platform for MNIST digit classification. It covers the full model lifecycle: data ingestion, training, serving, drift detection, active learning, and automated retraining — all running on Kubernetes.

## Architecture

<object data="fig/ml-system.drawio.pdf" type="application/pdf" width="100%" height="720" style="border-radius:4px;">
  <p><a href="fig/ml-system.drawio.pdf">Download architecture diagram (PDF)</a></p>
</object>

## How it works

Client requests arrive at an **Ingress Controller** and are forwarded to **Argo Rollout**, which handles traffic routing and zero-downtime model rollouts. **KEDA** autoscales the serving pods based on requests-per-second from Prometheus.

**Model Serving** runs the ONNX classifier and writes every prediction to Postgres and MinIO via the **Data Controller** facade. It also emits metrics scraped by **Alloy → Prometheus → Grafana**.

In the background, three services close the retraining loop:

- **Sampling Service** — pulls recent predictions from Postgres, scores them for annotation value, and marks candidates.
- **Annotation Service** — pulls marked candidates and writes ground-truth labels back.
- **Drift Monitoring** — watches Prometheus for distribution shift signals and new annotation counts.
- **Retrain Controller** — reads drift metrics and annotation counts; when thresholds are crossed it triggers an Argo Workflow that runs **Training**.

**Training** loads the versioned dataset from Postgres/MinIO via the Data Controller, trains a PyTorch model, and logs ONNX artifacts plus evaluation metrics through the **Model Artifact Controller** (backed by MLflow). A successful run triggers **Argo Rollout** to roll out the new model.

## Component legend

| Colour | Meaning |
|--------|---------|
| Blue border | Continuous service — runs as a Docker container, may be scaled |
| Green border | Facade — hides a backend, swappable without touching callers |
| Yellow border | Triggered or scheduled job |

## Key facades

| Facade | Hides | Used by |
|--------|-------|---------|
| [Data Controller](modules/data_controller.md) | Postgres + MinIO S3 | Serving, Training, Sampling, Annotation, Drift, Retrain |
| [Model Artifact Controller](modules/model_artifact_controller.md) | MLflow | Training, Serving |

## Quick start

For the full setup and test flow, see [Getting Started](getting-started/setup.md).

```bash
# One-time cluster setup
make k3d.create

# Build, import, and deploy all services
make k3d.build && make k3d.import && make k3d.deploy

# Prepare and seed the MNIST dataset
make data.setup

# Run training
make k3d.train
```
