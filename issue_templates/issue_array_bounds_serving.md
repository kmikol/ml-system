# Array Index Out of Bounds in Serving Prediction

**Labels**: `bug`, `high-priority`, `serving`

## Description

The `predict()` method directly indexes `logits[0]` and `embedding[0]` without validating that the arrays have the expected shape. This can cause an `IndexError` if the ONNX model returns an unexpected output shape.

## Affected Files

- `serving/main.py` (lines 150-154)

## Problem Details

Current implementation:

```python
def predict(self, features_array: np.ndarray) -> dict:
    with self._lock:
        if self.classifier_session is None:
            raise RuntimeError("Model not loaded")

        logits = self.classifier_session.run(
            None, {"input": features_array}
        )[0]

        embedding = self.embedder_session.run(
            None, {"input": features_array}
        )[0]

        probs = softmax(logits[0])  # Assumes logits has at least 1 element
        return {
            "probabilities": probs.tolist(),
            "embedding": embedding[0].tolist(),  # Assumes embedding[0] exists
        }
```

The code assumes:
1. `logits` has shape `(batch_size, num_classes)` with `batch_size >= 1`
2. `embedding` has shape `(batch_size, embedding_dim)` with `batch_size >= 1`

If the model returns an empty batch or unexpected shape, this causes an `IndexError`.

## Impact

- **Severity**: High
- **Likelihood**: Low (but possible with model changes or corrupted inputs)
- Can cause:
  - Application crash during prediction
  - Service unavailability
  - Failed health checks
  - Poor user experience

## Scenarios Where This Can Occur

1. Model export bug producing wrong output shape
2. ONNX runtime returning empty batches
3. Model architecture changes not reflected in serving code
4. Corrupted model files
5. Input preprocessing issues leading to empty batches

## Recommended Fix

Option 1: Validate array shapes (recommended)
```python
def predict(self, features_array: np.ndarray) -> dict:
    with self._lock:
        if self.classifier_session is None:
            raise RuntimeError("Model not loaded")

        logits = self.classifier_session.run(
            None, {"input": features_array}
        )[0]

        embedding = self.embedder_session.run(
            None, {"input": features_array}
        )[0]

        # Validate shapes
        if logits.shape[0] == 0:
            raise ValueError("Model returned empty logits batch")
        if embedding.shape[0] == 0:
            raise ValueError("Model returned empty embedding batch")

        probs = softmax(logits[0])
        return {
            "probabilities": probs.tolist(),
            "embedding": embedding[0].tolist(),
        }
```

Option 2: More defensive with detailed error messages
```python
def predict(self, features_array: np.ndarray) -> dict:
    with self._lock:
        if self.classifier_session is None:
            raise RuntimeError("Model not loaded")

        logits = self.classifier_session.run(
            None, {"input": features_array}
        )[0]

        embedding = self.embedder_session.run(
            None, {"input": features_array}
        )[0]

        # Validate shapes with detailed messages
        if len(logits.shape) != 2 or logits.shape[0] == 0:
            raise ValueError(
                f"Invalid logits shape: {logits.shape}. "
                f"Expected (batch_size, num_classes) with batch_size >= 1"
            )

        if len(embedding.shape) != 2 or embedding.shape[0] == 0:
            raise ValueError(
                f"Invalid embedding shape: {embedding.shape}. "
                f"Expected (batch_size, embedding_dim) with batch_size >= 1"
            )

        probs = softmax(logits[0])
        return {
            "probabilities": probs.tolist(),
            "embedding": embedding[0].tolist(),
        }
```

## Testing Recommendations

1. Add unit tests with:
   - Empty batch inputs
   - Models returning unexpected shapes
   - Corrupted model files
2. Add integration tests with various input shapes
3. Add property-based tests for shape invariants
4. Mock ONNX runtime to return edge case outputs
