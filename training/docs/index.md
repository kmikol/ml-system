# Training

The Training pipeline loads dataset versions, trains the classifier, exports ONNX artifacts, and registers model versions.

It logs metrics and reference distributions through the Model Artifact Controller facade.

## Design Decisions

- **[ADR-008: ONNX Model Export](../adr/008-onnx-model-export.md)** — Why models are exported to ONNX instead of PyTorch native format
- **[ADR-004: MLflow Artifact Storage](../adr/004-mlflow-artifact-storage.md)** — Model versioning and artifact management

## API (auto-generated)

::: training.main
