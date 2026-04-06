# ADR-008: ONNX Model Export Format Instead of PyTorch Native

**Date**: April 2026  
**Status**: Accepted  
**Deciders**: kmikol

## Context

The system trains PyTorch models but must choose a serialization format for storage and serving. Two main options were evaluated:

1. **PyTorch native format**: Save models as `.pt` files using `torch.save()` and load with `torch.load()`
2. **ONNX (Open Neural Network Exchange)**: Export models to `.onnx` format for framework-agnostic serving

The choice impacts:
- Deployment complexity and dependencies
- Framework lock-in risk
- Cross-platform portability
- Inference latency and performance

## Decision

The system exports trained PyTorch models to **ONNX format** before storage and serving.

- Training remains in PyTorch (leverages rich ecosystem: Lightning, torchvision, etc.)
- Models are exported to ONNX immediately after training validation
- Serving loads and runs inference via ONNX Runtime (lightweight, no PyTorch dependency)
- ONNX artifacts are the canonical model format stored in MLflow

## Rationale

### Framework Independence

ONNX is a cross-framework standard. A model exported to ONNX can theoretically be loaded and run by:
- ONNX Runtime (C++, Python, JavaScript, Java, etc.)
- TensorFlow (via ONNX converters)
- Other frameworks

### Lightweight Serving Dependencies

The serving container avoids PyTorch entirely:
- **PyTorch package**: includes CUDA libraries, model definitions, autograd graph
- **ONNX Runtime**: optimized inference engine only

This reduces:
- Deployment image size (faster pulls, smaller memory footprint)
- Cold-start time for serverless/Kubernetes pods


## Consequences

### Positive

- **Portable artifacts**: Models are framework-agnostic and can move between systems without retraining
- **Lightweight inference**: No PyTorch runtime in serving container

### Negative

- **Export complexity**: Not all PyTorch ops have direct ONNX equivalents; some models may not export cleanly
- **Debuggability**: ONNX is a lower-level IR; debugging inference bugs is harder than working with PyTorch
- **Versioning**: ONNX opset versions matter; old ONNX models may not load with newer ONNX Runtime


## Alternatives Considered

### Alternative 1: PyTorch Native Format (`.pt` files)

**Pros**: 
- Simple serialization (`torch.save()` / `torch.load()`)
- No export step, no compatibility issues
- Full debugging capability in PyTorch ecosystem

**Cons**: 
- Requires PyTorch installed in serving container
- Framework lock-in; models unportable to non-PyTorch systems
- Serving code tightly coupled to PyTorch internals

## Related Decisions

- **ADR-007** (ModelArtifactController): ONNX models are the canonical artifact format stored behind the artifact controller facade; the facade could theoretically support other formats (PyTorch, SavedModel) via adapters
- **ADR-004** (MLflow): MLflow stores ONNX artifacts alongside metadata (training hyperparams, evaluation metrics)

## Future Considerations

1. **Quantization for inference**: Post-export quantization (INT8, FLOAT16) could further reduce model size and latency; ONNX Runtime supports this natively
2. **Model validation framework**: Expand export validation to include performance benchmarks (latency, throughput) gated on acceptable ranges
3. **Custom operators**: If a model requires ops outside ONNX standard, evaluate custom operators or TVM compilation for efficient inference
4. **Mobile/edge deployment**: ONNX format enables easy export to mobile runtimes (ONNX Mobile, CoreML via conversion)
