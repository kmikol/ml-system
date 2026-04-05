# Architecture

## C4 Level 1: System Context

This level describes the system at high level: primary users, external systems, and the core MLOps platform boundary.

The `ml-system` platform serves model predictions and continuously adapts through monitoring, annotation, retraining, and rollout.

```mermaid
flowchart LR
                user[Client / Consumer]
                annotator[Human Annotator]

                subgraph platform[ML System Platform]
                        direction LR
                        serving[Serving API]
                        monitor[Monitoring and Drift Detection]
                        sample[Sampling Service]
                        annotation[Annotation Service]
                        training[Training Pipeline]
                        rollout[Model Rollout]
                end

                subgraph data_plane[Data and Artifact Plane]
                        postgres[(Postgres)]
                        minio[(MinIO S3)]
                        mlflow[(MLflow)]
                        metrics[(Prometheus)]
                        dashboards[Grafana]
                end

                user -->|predict| serving
                serving -->|prediction logs| postgres
                serving -->|sample payloads| minio
                serving -->|metrics| metrics

                monitor -->|reads metrics| metrics
                monitor -->|drift signal| sample
                sample -->|candidates| annotation
                annotator -->|labels| annotation
                annotation -->|annotations| postgres

                training -->|load training data| postgres
                training -->|load image arrays| minio
                training -->|register model artifacts| mlflow
                rollout -->|deploy promoted model| serving
                training --> rollout

                dashboards -->|visualize| metrics
```

## Scope and Intent

- Primary purpose: local-first MLOps experimentation with production-like control loops.
- Main user interaction: prediction requests to serving and operational monitoring via dashboards.
- Core feedback loop: drift detection -> annotation -> retraining -> model rollout.

## Next Levels

- C4 Level 2 (Container): service boundaries, protocols, and runtime responsibilities.
- C4 Level 3 (Component): internals of key services such as Serving, Data Controller, and Training.
