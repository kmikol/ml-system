# Missing Dictionary Key Check in ML Exporter

**Labels**: `bug`, `high-priority`, `monitoring`

## Description

The `_load_reference()` method in ML exporter directly accesses `data["prediction_class_frequencies"]` without checking if the key exists. This can raise a `KeyError` if the reference distribution artifact has a different format or is corrupted.

## Affected Files

- `monitoring/ml_exporter/main.py` (line 306)

## Problem Details

Current implementation:

```python
def _load_reference(self, run_id: str) -> list[float] | None:
    data = self._artifacts.download_reference_distribution(run_id, self._artifact_dir)
    freqs = data["prediction_class_frequencies"]  # <-- No "in" check
    return freqs if freqs else None
```

This assumes the artifact always contains the `prediction_class_frequencies` key, which may not be true if:
- Artifact format changes over time
- Different training runs produce different artifact formats
- Artifact is corrupted or incomplete
- Manual artifact uploads with different schema

## Impact

- **Severity**: High
- **Likelihood**: Medium (depends on artifact consistency)
- Can cause:
  - Monitoring service crash
  - Failed drift detection
  - Loss of monitoring metrics
  - Service downtime during model updates

## Recommended Fix

Option 1: Check for key existence (recommended)
```python
def _load_reference(self, run_id: str) -> list[float] | None:
    data = self._artifacts.download_reference_distribution(run_id, self._artifact_dir)

    if "prediction_class_frequencies" not in data:
        logger.warning(
            f"Missing 'prediction_class_frequencies' in reference "
            f"distribution for run {run_id}"
        )
        return None

    freqs = data["prediction_class_frequencies"]
    return freqs if freqs else None
```

Option 2: Use dict.get() with default
```python
def _load_reference(self, run_id: str) -> list[float] | None:
    data = self._artifacts.download_reference_distribution(run_id, self._artifact_dir)

    freqs = data.get("prediction_class_frequencies")
    if freqs is None:
        logger.warning(
            f"Missing 'prediction_class_frequencies' in reference "
            f"distribution for run {run_id}"
        )
        return None

    return freqs if freqs else None
```

Option 3: Validate artifact schema
```python
def _validate_reference_schema(self, data: dict) -> bool:
    """Validate that reference distribution has expected schema."""
    required_keys = ["prediction_class_frequencies"]
    return all(key in data for key in required_keys)

def _load_reference(self, run_id: str) -> list[float] | None:
    data = self._artifacts.download_reference_distribution(run_id, self._artifact_dir)

    if not self._validate_reference_schema(data):
        logger.error(
            f"Invalid schema in reference distribution for run {run_id}. "
            f"Expected keys: prediction_class_frequencies"
        )
        return None

    freqs = data["prediction_class_frequencies"]
    return freqs if freqs else None
```

## Additional Considerations

1. Document the expected artifact schema
2. Add schema validation in the training pipeline that creates the artifacts
3. Consider versioning the artifact format
4. Add monitoring for artifact format mismatches

## Testing Recommendations

1. Add unit tests with:
   - Missing key scenarios
   - Empty dictionary
   - Corrupted artifact data
   - Different artifact versions
2. Add integration tests with various artifact formats
3. Test backward compatibility with old artifacts
4. Mock artifact downloads returning unexpected formats
