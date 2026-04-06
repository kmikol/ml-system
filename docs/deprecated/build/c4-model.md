# C4 Architecture Model

## System Overview

The **ML System** is an end-to-end MLOps platform designed for learning production ML patterns in a local Kubernetes environment. It implements a closed-loop learning system: serve predictions → monitor for drift → detect data distribution shifts and anomalies → trigger retraining automatically → deploy updated models via canary rollout → scale based on demand. The system prioritizes pragmatism and local parity with production deployments.

**Key Purpose:**
- Serve MNIST predictions with low latency constraints (p99 < 1.0s)
- Detect batch drift via PSI (Population Stability Index) on class distributions
- Collect human annotations on flagged predictions to improve model performance
- Trigger automated retraining when data coverage and drift thresholds are met
- Perform safe canary deployments to validate new models before full rollout
- Auto-scale serving capacity based on real-time request arrival rate

---

## Level 1: System Context

```mermaid
graph TB
    Users["Users"]
    MLEngineers["ML Engineers"]
    Annotators["Annotators"]
    
    MLSystem["<b>Closed-Loop ML System</b><br/><br/>Kubernetes-native MLOps platform<br/>for automated model retraining,<br/>safe deployment, and drift monitoring"]
    
    Users -->|Prediction requests<br/>POST /predict| MLSystem
    MLSystem -->|Predictions + confidence| Users
    
    MLEngineers -->|Monitor metrics<br/>Review models<br/>Approve promotions| MLSystem
    MLSystem -->|Dashboards, alerts<br/>Model registry, status| MLEngineers
    
    Annotators -->|Label flagged samples| MLSystem
    MLSystem -->|Request annotations<br/>Sample predictions| Annotators
```

---

## Level 2: Container Diagram (Major Subsystems)

```mermaid
graph TB
    subgraph k3d["k3d Kubernetes Cluster"]
        subgraph Inference["Inference & Serving Subsystem"]
            Serving["Serving API<br/>(FastAPI)<br/>Handles /predict requests<br/>Enforces latency SLA<br/>Emits request metrics"]
            KEDA["KEDA Autoscaler<br/>Monitors request<br/>arrival rate<br/>Scales 1-15 replicas"]
        end
        
        subgraph Training["Training & Model Management"]
            TrainSvc["Training Service<br/>Trains on annotated<br/>predictions<br/>Evaluates new models"]
            MLflow["MLflow Registry<br/>Stores model versions<br/>Manages aliases<br/>(Prod/Canary/Staging)<br/>PostgreSQL backend"]
        end
        
        subgraph DataMgmt["Data Management & Storage"]
            Postgres["PostgreSQL<br/>Predictions database<br/>Annotation labels<br/>Metadata & MLflow backend"]
            MinIO["MinIO (S3)<br/>Model artifacts<br/>Training data<br/>Image samples"]
            Sampling["Sampling Service<br/>Selects predictions<br/>for annotation<br/>Marks candidates"]
            Annotation["Annotation Service<br/>Assigns ground truth<br/>labels from oracle<br/>Updates prediction records"]
        end
        
        subgraph MonitoringOrch["Monitoring & Orchestration"]
            Prometheus["Prometheus<br/>Scrapes metrics<br/>Stores time series<br/>Evaluates alert rules<br/>3 alerts: PSI, Latency, Annotation count"]
            Events["Argo Events<br/>Correlates multi-alert<br/>conditions<br/>Triggers workflows<br/>via NATS EventBus"]
            Workflows["Argo Workflows<br/>Orchestrates<br/>sample-and-label<br/>retrain pipelines<br/>Manages canary promotion"]
            MonitorExp["Monitoring Exporter<br/>Queries predictions<br/>Computes drift metrics<br/>Exposes to Prometheus<br/>(PSI, class dist, latency)"]
        end
        
        Serving -->|Store predictions| Postgres
        Serving -->|Save images| MinIO
        Serving -->|Emit predict_arrivals_total| Prometheus
        
        KEDA -->|Query request arrival rate| Prometheus
        KEDA -->|Scale replicas up/down| Serving
        
        TrainSvc -->|Load annotated data| Postgres
        TrainSvc -->|Save trained model| MLflow
        TrainSvc -->|Store artifacts| MinIO
        
        MLflow -->|Metadata storage| Postgres
        MLflow -->|Model artifacts| MinIO
        
        Sampling -->|Query predictions| Postgres
        Sampling -->|Mark for annotation| Postgres
        
        Annotation -->|Fetch marked samples| Postgres
        Annotation -->|Update with labels| Postgres
        
        MonitorExp -->|Query all predictions| Postgres
        MonitorExp -->|Expose metrics| Prometheus
        
        Prometheus -->|PSI > 0.25| Events
        Prometheus -->|Annotation count >= 50| Events
        Prometheus -->|Latency p99 > 1.0s| Events
        
        Events -->|Trigger workflow| Workflows
        
        Workflows -->|Execute| Sampling
        Workflows -->|Execute| Annotation
        Workflows -->|Execute| TrainSvc
        Workflows -->|Load & promote models| MLflow
    end
    
    Users -->|POST /predict| Serving
    Serving -->|Predictions| Users
    
    MLEngineers -->|Monitor via dashboards| Prometheus
    MLEngineers -->|Review & approve| MLflow
    MLEngineers -->|Manage workflows| Workflows
    
    Annotators -->|sampled predictions| Sampling
    Annotators -->|label submissions| Annotation
```

---

## Subsystem Descriptions

### Inference & Serving Subsystem
Handles real-time prediction requests from users. The Serving API receives images via REST and returns predictions with confidence scores. KEDA monitors request arrival rates and auto-scales serving pods (1–15 replicas) to maintain performance under load. Metrics (predict_arrivals_total) are emitted to Prometheus for scaling decisions.

### Training & Model Management
Manages the model lifecycle. The Training Service operates on-demand (triggered by Argo Workflows) to retrain models on newly annotated data. MLflow maintains a model registry with version control and alias management (Production for stable, Canary for validation). Models are stored as ONNX artifacts with reference distributions for drift detection.

### Data Management & Storage
Persistent storage layer. PostgreSQL holds prediction records with metadata, annotation labels, and MLflow backend state. MinIO (S3-compatible) stores model artifacts, training data, and image samples. The Sampling Service marks predictions for human annotation; the Annotation Service assigns ground truth labels from an oracle, simulating the feedback loop.

### Monitoring & Orchestration
Implements event-driven retraining logic. Prometheus scrapes metrics (drift PSI, latency, confidence) and evaluates 3 alert rules. Argo Events correlates multi-condition alerts (e.g., drift + data availability) via NATS EventBus and triggers Argo Workflows. Workflows orchestrate the data sampling → annotation → retraining → canary promotion pipeline. The Monitoring Exporter continuously computes drift metrics and exposes them for alerting.

---

## Key Flows

- **Inference Flow**: User → Serving API → PostgreSQL (store) + Prometheus (metrics) → Response
- **Drift Detection**: Monitoring Exporter → Queries predictions → Computes PSI → Prometheus → Alert → Argo Events → Workflow
- **Retraining**: Alert triggers → Argo Workflows → Sampling → Annotation → Training → MLflow (new version) → Canary promotion
- **Autoscaling**: KEDA → Queries predict_arrivals_total → Scales Serving replicas based on RPS target (5 per replica)

