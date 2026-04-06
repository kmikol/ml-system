# Serving

The Serving service hosts the inference API, loads model artifacts (in ONNX format), and persists prediction records.

It also publishes operational metrics for autoscaling and monitoring.

## Design Decisions

- **[ADR-008: ONNX Model Export](../adr/008-onnx-model-export.md)** — Why models are loaded from ONNX format instead of PyTorch native format
- **[ADR-002: Canary Rollouts](../adr/002-canary-rollouts.md)** — Safe model promotion strategy via Argo Rollouts
- **[ADR-006: KEDA Autoscaling](../adr/006-keda-autoscaling.md)** — Request-rate driven pod autoscaling

## API (auto-generated)

::: serving.main
