# Architecture

See the [home page](index.md) for the full diagram.

## Request path

```
Client
  └─▶ Ingress Controller   (routes to cluster)
        └─▶ Argo Rollout    (traffic routing + model rollout)
              └─▶ Model Serving   (ONNX inference)
                    ├─▶ Data Controller ──▶ Postgres  (prediction record)
                    │                  └─▶ MinIO S3   (image bytes)
                    └─▶ Logging ──▶ Alloy ──▶ Prometheus ──▶ Grafana
```

KEDA watches the Prometheus requests/sec metric and scales the serving pods up or down accordingly.

## Retraining loop

```
Drift Monitoring  ◀── Prometheus (drift signals)
                  ◀── Data Controller (new annotation counts)
                    │
                    ▼
             Retrain Controller
                    │  (threshold crossed)
                    ▼
             Retrain Trigger ──▶ Training
                                    ├─▶ Data Controller  (load dataset)
                                    └─▶ Model Artifact Controller ──▶ MLflow
                                                                        │
                                                              new model registered
                                                                        │
                                                                        ▼
                                                               Argo Rollout
```

## Annotation loop

```
Model Serving ──▶ Data Controller ──▶ Postgres (predictions with annotation_status='none')
                                          │
                                          ▼
                                  Sampling Service
                                  (scores candidates, marks annotation_status='candidate')
                                          │
                                          ▼
                                  Annotation Service  ◀── Annotate Trigger
                                  (human labels, writes annotation_status='annotated')
                                          │
                                          ▼
                                  Data Controller ──▶ Postgres (label written)
```

## Facades

Two green-bordered facades isolate backend implementation details from the rest of the system.

### Data Controller

Hides **Postgres** (prediction records, dataset metadata) and **MinIO S3** (image bytes). Each service gets a role-scoped subclass:

| Subclass | Service | Operations |
|----------|---------|------------|
| `ServingDataController` | Model Serving | `store_prediction` |
| `DriftDataController` | Drift Monitoring | `get_predictions`, `get_labeled_predictions` |
| `SamplingDataController` | Sampling Service | `get_predictions`, `mark_candidate`, `count_labels_since` |
| `AnnotationDataController` | Annotation Service | `write_label` |
| `DatasetController` | Training | `store_sample`, `get_dataset_split` |
| `FakeDataController` | Unit tests | Full surface, in-memory |

### Model Artifact Controller

Hides **MLflow**. Exposes a `ModelArtifactController` Protocol so the backend can be swapped without touching any caller:

| Operation | Used by |
|-----------|---------|
| `start_run` / `log_params` / `log_metrics` / `log_artifacts` | Training |
| `register_model` / `promote_model` | Training (post-run) |
| `get_production_run_id` / `download_artifacts` | Model Serving |

## Storage layout

```
Postgres
├── predictions        — one row per inference (image, embedding, prediction, label, status)
└── dataset_samples    — one row per training sample (metadata + MinIO path)

MinIO S3
└── mnist-dataset/
    └── {date}/{uuid}.npy    — float32 image arrays

MLflow artifacts (per training run)
└── onnx/
    ├── classifier/model.onnx
    └── embedder/model.onnx
    reference_distribution.json
    class_gaussians.json
    feature_schema.json
```

## Infrastructure

| Component | Role |
|-----------|------|
| k3d / k3s | Local Kubernetes cluster |
| Helm (`helm/ml-system`) | Service deployment |
| KEDA | Horizontal pod autoscaling via Prometheus metrics |
| Argo Rollout | Progressive model delivery |
| Alloy | Log and metric collection agent |
| Prometheus | Metrics store |
| Grafana | Dashboards |
