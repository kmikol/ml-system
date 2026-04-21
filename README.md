# ml-system

A production-grade MLOps platform demonstrating end-to-end machine learning infrastructure:
model serving, continuous retraining, and safe canary deployments — running locally on Kubernetes.

**[Documentation](https://kmikol.github.io/ml-system/architecture/)**,
**[Getting Started](https://kmikol.github.io/ml-system/getting-started/setup/)**,
**[Open Issues](https://github.com/kmikol/ml-system/issues)**

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://kmikol.github.io/ml-system/fig/ml-system.dark.drawio.svg">
  <img src="https://kmikol.github.io/ml-system/fig/ml-system.light.drawio.svg" alt="ML System Architecture">
</picture>

---

## What it does

- **Serves predictions** via a FastAPI inference server with ONNX runtime, scaled horizontally by KEDA based on live request metrics
- **Retrains continuously** through an event-driven Argo Workflows pipeline: sample → annotate → train → evaluate
- **Deploys safely** via Argo Rollouts canary deployments with automated PSI drift analysis gating promotion
- **Abstracts storage** behind facade services — application code never couples directly to Postgres, MinIO, or MLflow
- **Observes everything** with Prometheus, Grafana, and Loki across all services

Current implementation: MNIST classification end-to-end. Architecture is domain-agnostic.

---

## Stack

Kubernetes (k3s) · Helm · FastAPI · ONNX · Argo Workflows · Argo Rollouts · KEDA · MLflow · Postgres · MinIO · Prometheus · Grafana · Loki

---

## [Quickstart](https://kmikol.github.io/ml-system/getting-started/setup/)

---

## Documentation

Full architecture, design decisions, and module references live in the docs site — the README
intentionally stays lean.

| | |
|---|---|
| Architecture overview | [kmikol.github.io/ml-system/architecture](https://kmikol.github.io/ml-system/architecture/) |
| Getting started | [kmikol.github.io/ml-system/getting-started/setup](https://kmikol.github.io/ml-system/getting-started/setup/) |
| ADRs (8 decisions) | [kmikol.github.io/ml-system/adr](https://kmikol.github.io/ml-system/adr/) |
| Module reference | [kmikol.github.io/ml-system/modules](https://kmikol.github.io/ml-system/modules/) |
| API & event schemas | [kmikol.github.io/ml-system/schemas](https://kmikol.github.io/ml-system/schemas/) |
