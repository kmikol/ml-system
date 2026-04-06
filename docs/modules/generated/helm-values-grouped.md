# Helm Values (Grouped)

Auto-generated from `helm/ml-system/values.yaml`.
This view keeps values grouped by top-level section with nested subsection tables.

## global

Global defaults shared across chart templates.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| namespace | `ml-system` | Kubernetes namespace used by chart resources. |
| imageRegistry | `""` | Optional container registry prefix for images. Empty keeps image names as-is. |
| imagePullPolicy | `IfNotPresent` | Default image pull policy for custom images. |

## serving

FastAPI serving deployment and service configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable serving deployment. |
| replicas | `1` | Number of serving replicas when autoscaling is disabled or at baseline. |
| image | `ml-system-serving:latest` | Serving container image reference. |
| port | `8000` | Container port exposed by the serving application. |

### env

| Key | Default | Description |
|-----|---------|-------------|
| DATA_CONTROLLER_DB_URL | `"postgresql://mlflow:mlflow@postgres:5432/mlflow"` | PostgreSQL DSN used by Data Controller facade. |
| LOKI_URL | `"http://loki:3100"` | Loki endpoint for structured log shipping. |
| DATASET_S3_ENDPOINT_URL | `"http://minio:9000"` | S3-compatible endpoint used for dataset object access. |
| DATASET_BUCKET | `"mnist-dataset"` | Dataset bucket name for sample/object retrieval. |

### resources

| Key | Default | Description |
|-----|---------|-------------|
| requests | `{ memory: "512Mi", cpu: "500m" }` | Resource requests for serving pods. |
| limits | `{ memory: "1Gi", cpu: "1000m" }` | Resource limits for serving pods. |

### service

| Key | Default | Description |
|-----|---------|-------------|
| type | `ClusterIP` | Service type for serving traffic. |

## mlflow

MLflow deployment and service configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable MLflow deployment. |
| image | `ml-system-mlflow:latest` | MLflow container image reference. |
| port | `5000` | Container port exposed by MLflow. |
| backendUri | `"postgresql://mlflow:mlflow@postgres:5432/mlflow"` | SQLAlchemy backend URI for MLflow metadata store. |
| artifactRoot | `"s3://mlflow-artifacts/"` | Default artifact root URI for MLflow runs. |
| s3EndpointUrl | `"http://minio:9000"` | S3-compatible endpoint used by MLflow artifact store. |

### service

| Key | Default | Description |
|-----|---------|-------------|
| type | `ClusterIP` | Service type for MLflow endpoint. |

## postgres

PostgreSQL stateful service configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable PostgreSQL deployment. |
| image | `postgres:15` | PostgreSQL container image. |
| db | `mlflow` | Database name created for system components. |
| user | `mlflow` | Database user name for system components. |

### persistence

| Key | Default | Description |
|-----|---------|-------------|
| size | `5Gi` | PVC size for PostgreSQL data. |

## minio

MinIO object storage configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable MinIO deployment. |
| image | `minio/minio` | MinIO server image. |
| mcImage | `minio/mc` | MinIO client image used by helper jobs/init tasks. |

### persistence

| Key | Default | Description |
|-----|---------|-------------|
| size | `10Gi` | PVC size for MinIO object data. |

### service

| Key | Default | Description |
|-----|---------|-------------|
| type | `ClusterIP` | Service type for MinIO endpoints. |

## lakefs

lakeFS deployment and object-versioning configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable lakeFS deployment. |
| image | `treeverse/lakefs:1.48` | lakeFS container image reference. |
| port | `8000` | Container port exposed by lakeFS API. |
| repo | `ml-system-datasets` | lakeFS repository name used for dataset versioning. |
| storageNamespace | `"s3://lakefs-data/"` | Underlying object-store namespace for lakeFS. |

### service

| Key | Default | Description |
|-----|---------|-------------|
| type | `ClusterIP` | Service type for lakeFS endpoint. |

## prometheus

Prometheus monitoring stack configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable Prometheus deployment. |
| image | `prom/prometheus:v2.50.0` | Prometheus container image reference. |
| port | `9090` | Container port exposed by Prometheus. |
| scrapeIntervalSeconds | `15` | Scrape interval in seconds for target collection. |
| evaluationIntervalSeconds | `15` | Rule evaluation interval in seconds. |

### persistence

| Key | Default | Description |
|-----|---------|-------------|
| size | `5Gi` | PVC size for Prometheus TSDB storage. |

### service

| Key | Default | Description |
|-----|---------|-------------|
| type | `ClusterIP` | Service type for Prometheus endpoint. |

## alertmanager

Alertmanager configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable Alertmanager deployment. |
| image | `prom/alertmanager:v0.26.0` | Alertmanager container image reference. |
| port | `9093` | Container port exposed by Alertmanager. |

## grafana

Grafana dashboard service configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable Grafana deployment. |
| image | `grafana/grafana:10.3.0` | Grafana container image reference. |
| port | `3000` | Container port exposed by Grafana. |
| adminPassword | `admin` | Grafana admin password for local/dev use. |

### persistence

| Key | Default | Description |
|-----|---------|-------------|
| size | `1Gi` | PVC size for Grafana storage. |

### service

| Key | Default | Description |
|-----|---------|-------------|
| type | `ClusterIP` | Service type for Grafana endpoint. |

## alloy

Grafana Alloy metrics/logs collector configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable Alloy deployment. |
| image | `grafana/alloy:latest` | Alloy container image reference. |
| scrapeIntervalSeconds | `15` | TODO: confirm whether this drives scrape config directly or template defaults only. |

## loki

Loki log storage/query configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable Loki deployment. |
| image | `grafana/loki:2.9.0` | Loki container image reference. |
| port | `3100` | Container port exposed by Loki. |

### persistence

| Key | Default | Description |
|-----|---------|-------------|
| size | `5Gi` | PVC size for Loki chunks/index. |

### service

| Key | Default | Description |
|-----|---------|-------------|
| type | `ClusterIP` | Service type for Loki endpoint. |

## autoscaling

KEDA-based serving autoscaling settings.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `true` | Enable/disable KEDA autoscaling resources for serving rollout. |
| minReplicas | `1` | Minimum number of serving replicas. |
| maxReplicas | `10` | Maximum number of serving replicas. |
| targetRPS | `5` | Target requests/second per replica; KEDA scales to ceil(total_rps / targetRPS). |
| rateWindow | `"30s"` | Rate window used in Prometheus query (recommended >= 2x scrape interval). |
| pollingIntervalSeconds | `10` | KEDA polling interval for external metrics. |
| cooldownPeriod | `30` | Cooldown period before scaling actions settle. |
| scaleUpPercent | `100` | Maximum percentage increase per scale-up period. |
| scaleUpPeriodSeconds | `15` | Scale-up period in seconds for growth policy. |
| scaleDownStabilizationSeconds | `60` | Stabilization window in seconds before scale-down takes effect. |
| scaleDownPeriodSeconds | `15` | Scale-down period in seconds for reduction policy. |

## secrets

Shared credentials and secrets (override in production).

### Values

| Key | Default | Description |
|-----|---------|-------------|
| minioAccessKey | `minioadmin` | MinIO access key used by in-cluster clients. |
| minioSecretKey | `minioadmin` | MinIO secret key used by in-cluster clients. |
| postgresPassword | `mlflow` | PostgreSQL password for configured user. |

### lakefs

| Key | Default | Description |
|-----|---------|-------------|
| accessKeyId | `AKIAIOSFODNN7EXAMPLE` | lakeFS access key ID for object-store auth. |
| secretAccessKey | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | lakeFS secret access key for object-store auth. |
| encryptSecretKey | `"change-me-in-production"` | Secret used by lakeFS for encryption-at-rest features. |

## mlExporter

ML exporter deployment and service configuration.

### Values

| Key | Default | Description |
|-----|---------|-------------|
| enabled | `false` | Enable/disable ML exporter deployment. |
| image | `ml-system-ml-exporter:latest` | ML exporter container image reference. |
| port | `8001` | Container port exposed by ML exporter. |

### env

| Key | Default | Description |
|-----|---------|-------------|
| DATA_CONTROLLER_DB_URL | `"postgresql://mlflow:mlflow@postgres:5432/mlflow"` | PostgreSQL DSN used by exporter Data Controller. |
| MLFLOW_TRACKING_URI | `"http://mlflow:5000"` | MLflow tracking endpoint for model/artifact reads. |
| MODEL_NAME | `"ml_system_model"` | Registered model name used for reference artifact lookup. |
| MLFLOW_S3_ENDPOINT_URL | `"http://minio:9000"` | S3-compatible endpoint for MLflow artifact access. |
| DRIFT_POLL_INTERVAL | `"5"` | Poll interval in seconds for drift export loop. |
| DRIFT_WINDOW_SECONDS | `"300"` | Window size in seconds used for drift calculations. |
| REFERENCE_CACHE_TTL_SECONDS | `"300"` | TTL in seconds for cached reference artifacts. |

### resources

| Key | Default | Description |
|-----|---------|-------------|
| requests | `{ memory: "256Mi", cpu: "100m" }` | Resource requests for ML exporter pods. |
| limits | `{ memory: "512Mi", cpu: "250m" }` | Resource limits for ML exporter pods. |

### service

| Key | Default | Description |
|-----|---------|-------------|
| type | `ClusterIP` | Service type for ML exporter endpoint. |
