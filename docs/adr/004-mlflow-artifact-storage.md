# ADR-004: MLflow Model Registry with PostgreSQL + MinIO Backend

**Date**: 2024-03-29  
**Status**: Accepted  
**Deciders**: ML Platform Team

## Context

The system must manage trained models across multiple training iterations, tracking versions, assigning roles (stable vs. candidate), and enabling serving pods to load the correct model variant at runtime. Requirements:

1. **Versioning**: Each training run produces a new model; history must be queryable
2. **Aliases for routing**: Serving needs to load different models based on pod role (Production stable vs. Canary candidate)
3. **Production parity**: Development environment should simulate production-grade model management, not oversimplified local storage
4. **Portability**: Model registry interface should support future backend swaps (cloud ML platforms, alternative registries)

Two approaches were evaluated:

1. **Simplified filesystem + custom logic**: Store models as timestamped files; implement custom alias/versioning logic in Python
2. **Model registry (MLflow)**: Use MLflow's Model Registry for versioning, metadata, and alias management with pluggable storage backend

## Decision

The system uses **MLflow Model Registry** with:

- **Metadata store**: PostgreSQL (`postgresql://mlflow:mlflow@postgres:5432/mlflow`)
- **Artifact store**: MinIO S3-compatible (`s3://mlflow-artifacts/`)
- **Abstraction layer**: `ModelArtifactController` facade (Python) allows future backend swaps without code changes to training/serving

```
Training
  ↓
MLflow Client (logs artifacts + parameters)
  ↓
MLflow Server (port 5000)
  ├─ PostgreSQL: stores run metadata, versions, aliases
  └─ MinIO S3: stores ONNX models + reference distributions + Gaussians
  
Serving
  ↓
ModelArtifactController (abstraction facade)
  ↓
MLflow Client: get_model_version_by_alias(Production/Canary)
  ↓
MinIO: download ONNX + reference artifacts
```

**Model aliases for traffic routing**:
- `Production`: Current stable model (100% traffic to stable replica set)
- `Canary`: Candidate model under evaluation (0-100% traffic during canary rollout)
- `Staging`: Reserved for future use (e.g., A/B testing)

**Model artifact structure per run**:
```
{run_id}/
├── onnx/classifier/model.onnx          ← Neural network (14 classes)
├── onnx/embedder/model.onnx            ← Feature extractor
├── reference_distribution.json         ← Baseline class frequencies
├── class_gaussians.json                ← μ, Σ per class for Mahalanobis
└── feature_schema.json                 ← Input validation schema
```

## Rationale

**Pragmatic tooling choice**: MLflow was selected for development velocity and familiarity. Rather than building custom versioning + alias logic, MLflow provides production-grade features (API, UI, atomic operations) out of the box. The system is designed to be trainable by practitioners unfamiliar with MLflow; using an industry-standard tool ensures skills transfer.

**Production environment simulation**: Using PostgreSQL + MinIO (not SQLite + local filesystem) means the development environment mimics production deployment patterns:
- Multi-client concurrent access (multiple workflows writing simultaneously)
- Remote artifact storage (requires network calls, mirrors cloud practice)
- Transactional consistency (PostgreSQL ensures atomicity of version/alias updates)

This teaches practices that don't transfer when using simplified local storage.

**Alias-based model loading**: Instead of hardcoding model versions in Kubernetes or serving code, aliases allow operational changes without redeployment. Serving polls MLflow every 10 seconds; when the Production alias changes, the next prediction loads the new model. This enables:
- Canary deployments: set Canary alias to new version, Argo Rollout shifts traffic gradually
- Quick rollbacks: if issues emerge, immediately reassign Production alias to previous version
- Decoupling: training workflow and serving don't need to know each other's infrastructure

**Reference distributions and Gaussians precomputed**: The `reference_distribution.json` (baseline class frequencies) and `class_gaussians.json` (per-class μ and Σ) are computed during training for two reasons:
- **Drift detection**: Serving compares live prediction distribution against baseline; PSI metric quantifies shift
- **Outlier detection**: Mahalanobis distance per sample requires reference covariance; computing per-request would be slow and require dataset access

Storing these with the model ensures:
- No data leakage (reference data is tied to the model version used to compute it)
- Consistent evaluation (production and canary use baseline from their respective training run)
- Reproducibility (exact PSI thresholds depend on baseline; versioning baselines aligns with model versions)

**ModelArtifactController abstraction facade**: The serving and training code reference a facade, not MLflow directly. This enables future swaps:

```python
# Training writes via controller
controller = ModelArtifactController()
controller.start_run()
controller.log_artifacts(...)
controller.register_model(...)
controller.set_alias(alias="Production", version=...)

# Serving reads via controller
controller = ModelArtifactController()
run_id = controller.get_run_by_alias("Production")
onnx_path = controller.download_artifacts(run_id)
```

If a superior registry emerges (e.g., cloud-native managed services, Hugging Face Model Hub), swapping backends requires changes only to the controller implementation, not to application code.

## Consequences

**Positive**:
- Full versioning history: query all prior models, parameters, metrics with MLflow APIs
- Alias-based safety: can immediately rollback by assigning Production alias to previous version
- Multi-client concurrent safe: PostgreSQL handles Atlas-like conflicts, no custom locking needed
- Production-grade simulation: using real backends teaches transferable patterns
- Queryable metadata: MLflow UI enables ad-hoc exploration of runs, comparison of hyperparameters
- Standardized interface: team can leverage MLflow knowledge from other projects

**Negative**:
- Infrastructure overhead: PostgreSQL + MLflow server + MinIO must be running and healthy
- Dependency coupling: training, serving, and workflows all depend on MLflow availability
- Latency: Serving polls MLflow every 10 seconds (up to ~15s delay between alias change and model load)
- Artifact storage size: Without cleanup, S3 bucket accumulates hundreds of model versions
- Debugging complexity: Model failures require tracing through MLflow server logs, PostgreSQL, etc.

**Operational requirements**:
- Regular backups of PostgreSQL (contains all model history and alias assignments)
- Monitoring MLflow server health (pod crashes mean serving can't load new models)
- S3 bucket cleanup policy (old versions should be deleted after promotion/retention window)
- Credentials management (MinIO access keys, PostgreSQL passwords in Kubernetes secrets)

## Alternatives Considered

### Alternative 1: Custom Model Storage (Filesystem)
Store models as timestamped directories in shared PVC or MinIO with custom Python versioning logic.

**Pros**: 
- Simple, no MLflow server needed
- Direct control over storage layout

**Cons**: 
- No built-in versioning API (must implement query logic)
- No alias support (must hard-code model paths in serving code)
- Doesn't scale to concurrent multi-model management
- Versioning mistakes are harder to detect
- Not production-like

### Alternative 2: Only S3 (No MLflow Registry)
Store models directly in S3 with paths like `s3://models/v1/classifier.onnx`.

**Pros**: 
- Minimal overhead (just cloud storage)
- Works with existing CI/CD tools

**Cons**: 
- Metadata is unstructured (no versioning API, no experiment tracking)
- Alias support requires custom code (serving must poll S3 directory for "latest" pointer)
- No audit trail of model promotion decisions
- Harder to coordinate between training and serving

### Alternative 3: Alternative Registry (BentoML, Seldon)
Use a different model management platform designed for serving integration.

**Pros**: 
- May have tighter serving integration
- Different trade-offs on metadata vs. simplicity

**Cons**: 
- Adds learning curve for team
- Ecosystem may be smaller than MLflow
- Still requires external storage backend (S3, database) anyway
- Not necessarily simpler than MLflow

## Implementation Details

**Training Model Registration** (training/main.py):

```python
from shared.model_artifact_controller import ModelArtifactController

controller = ModelArtifactController()
with controller.start_run() as run_id:
    # Train model
    classifier = train_classifier(train_data)
    embedder = extract_features(train_data)
    
    # Log artifacts
    controller.log_model(classifier, "onnx/classifier")
    controller.log_model(embedder, "onnx/embedder")
    
    # Compute and log reference baseline
    ref_dist = compute_reference_distribution(train_data)
    controller.log_artifact_dict(ref_dist, "reference_distribution.json")
    
    # Compute Gaussians for Mahalanobis distance
    gaussians = fit_class_gaussians(train_data, classifier)
    controller.log_artifact_dict(gaussians, "class_gaussians.json")
    
    # Register
    version = controller.register_model(model_name=MODEL_NAME)
    # version is registered but NOT aliased yet; evaluation/promotion decides
```

**Serving Model Loading** (serving/main.py):

```python
from shared.model_artifact_controller import ModelArtifactController

class ModelManager:
    def load_from_mlflow(self):
        """Poll MLflow every SERVING_MODEL_POLL_INTERVAL seconds"""
        stage = "Production" if SERVING_ROLE == "stable" else "Canary"
        
        # Get current alias (returns None if alias not set)
        run_id = self.controller.get_run_by_alias(model_name, stage=stage)
        if run_id == self._cached_run_id:
            return True  # Already loaded
        
        # Download artifacts
        bundle = self.controller.download_serving_bundle(run_id)
        classifier_path = bundle["classifier"]
        embedder_path = bundle["embedder"]
        gaussians = bundle["gaussians"]
        
        # Load ONNX sessions atomically
        with self._load_lock:
            self.classifier_session = ort.InferenceSession(classifier_path)
            self.embedder_session = ort.InferenceSession(embedder_path)
            self.class_gaussians = gaussians
            self._cached_run_id = run_id
        
        return True
```

**Model Promotion (Retrain Workflow)**, step-by-step:

```yaml
# retrain-workflow (k8s/argo/workflows/retrain.yaml)

# Step 3: Assign Canary alias to new model
- name: set-canary-alias
  script:
    command: [python]
    source: |
      from shared.model_artifact_controller import ModelArtifactController
      controller = ModelArtifactController()
      # This call is atomic; MLflow ensures only one version holds Canary
      controller.set_alias(
        model_name=MODEL_NAME,
        alias="Canary",
        version=TRAINED_VERSION
      )
      # ml_exporter now monitors Canary PSI
      # Serving (canary RS) loads Canary model on next poll

# Step 6: Promote to Production on canary success
- name: promote-production
  script:
    command: [python]
    source: |
      controller = ModelArtifactController()
      controller.set_alias(
        model_name=MODEL_NAME,
        alias="Production",
        version=TRAINED_VERSION
      )
      # Note: MLflow auto-clears old Production alias
      # Serving (stable RS) loads new Production model in ~10-15s
```

## Related Decisions

- **ADR-002**: Canary rollouts use Argo Rollouts with different pod labels (stable vs. canary); each loads a different alias from MLflow
- **ADR-005**: ML Exporter polls MLflow to know which model version is active (stable vs. canary)

## Future Considerations

1. **Model cleanup**: Implement retention policy (delete versions older than 30 days, keep last 5)
2. **A/B testing**: Add more aliases (CandidateA, CandidateB) for multi-model comparison
3. **Model compression**: Archive ONNX models to cheaper storage tier after stable period
4. **Feature store integration**: Link models to feature schemas; track schema evolution per version
5. **MLflow Recipes**: Use MLflow's higher-level abstraction if moving to production
6. **Cross-environment promotion**: Script to export staging registry, import to production
