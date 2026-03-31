# Model Manager Thread Safety Issue During Inference

**Labels**: `bug`, `critical`, `concurrency`, `serving`

## Description

The `predict()` method in the ModelManager class has a TOCTOU (Time-of-check to time-of-use) race condition. The lock is released after checking that the model is loaded, but before the actual inference runs. This means the model sessions could be replaced by another thread during inference.

## Affected Files

- `serving/main.py` (lines 137-158)

## Problem Details

The current implementation acquires a lock, checks if the model is loaded, then releases the lock before running inference:

```python
def predict(self, features_array: np.ndarray) -> dict:
    with self._lock:
        if self.classifier_session is None:
            raise RuntimeError("Model not loaded")
        logits = self.classifier_session.run(...)  # Line 141-144
        embedding = self.embedder_session.run(...)  # Line 145-148
    # Lock is released here, but inference happens inside the with block
```

The issue is that while the lock is held during the `run()` calls, another thread could still call `load_model()` which would replace `self.classifier_session` and `self.embedder_session`. This creates a race condition where:

1. Thread A starts inference with model version 1
2. Thread B calls `load_model()` and replaces sessions with version 2
3. Thread A continues inference, but might now be using version 2 sessions
4. Results are inconsistent (logits from v1, embeddings from v2, or vice versa)

## Impact

- **Severity**: Critical
- **Likelihood**: Medium (depends on model reload frequency)
- Can cause:
  - Inconsistent predictions (mixing embeddings/logits from different models)
  - Potential crashes if sessions are invalidated during inference
  - Race conditions during model updates
  - Unpredictable behavior when models are reloaded

## Recommended Fix

Option 1: Hold the lock for the entire inference duration
```python
def predict(self, features_array: np.ndarray) -> dict:
    with self._lock:
        if self.classifier_session is None:
            raise RuntimeError("Model not loaded")

        # Hold lock during entire inference
        logits = self.classifier_session.run(
            None, {"input": features_array}
        )[0]

        embedding = self.embedder_session.run(
            None, {"input": features_array}
        )[0]

        probs = softmax(logits[0])
        return {
            "probabilities": probs.tolist(),
            "embedding": embedding[0].tolist(),
        }
```

Option 2: Create local references under lock
```python
def predict(self, features_array: np.ndarray) -> dict:
    with self._lock:
        if self.classifier_session is None:
            raise RuntimeError("Model not loaded")
        # Create local references while holding lock
        classifier = self.classifier_session
        embedder = self.embedder_session

    # Use local references (safe from replacement)
    logits = classifier.run(None, {"input": features_array})[0]
    embedding = embedder.run(None, {"input": features_array})[0]
    ...
```

Option 3: Use RWLock for better concurrency
```python
# Allow multiple readers (predict) but exclusive writer (load_model)
from threading import RLock
import contextlib

class ModelManager:
    def __init__(self):
        self._lock = RLock()
        # ... rest of init ...
```

## Testing Recommendations

1. Add concurrency tests that:
   - Run multiple predictions simultaneously
   - Reload models during inference
   - Verify prediction consistency
2. Add stress tests with concurrent model loads and predictions
3. Use thread sanitizers during testing
