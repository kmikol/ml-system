# Model Artifact Controller

Facade that hides **MLflow** from the rest of the system. No module outside `shared/model_artifact_controller.py` imports `mlflow` directly.

The `ModelArtifactController` Protocol defines the interface. `MLflowModelArtifactController` is the production implementation. Swap backends by writing a new class that satisfies the Protocol — nothing else changes.

All MLflow errors are wrapped in `ModelArtifactError`.

## shared.model_artifact_controller

::: shared.model_artifact_controller
