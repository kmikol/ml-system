## PredictRequest

### Fields

| Field | Type | Description |
|------|------|-------------|
| image | list[list[float]] | 14x14 grayscale image, values in [0, 1] |
| uuid | UUID \| None | UUID of this sample, if known (enables annotation pipeline) |

## PredictResponse

### Fields

| Field | Type | Description |
|------|------|-------------|
| prediction | int | Predicted class index. |
| confidence | float | Confidence score for predicted class. |
| model_version | str | Model version used to generate prediction. |
| uuid | UUID | UUID associated with this prediction record. |

## ValidationErrorResponse

### Fields

| Field | Type | Description |
|------|------|-------------|
| detail | str | High-level error summary. |
| errors | list[dict] | Detailed validation errors. |

## HealthResponse

### Fields

| Field | Type | Description |
|------|------|-------------|
| status | str | Service health state, e.g. healthy/unhealthy. |
| model_loaded | bool | Whether model is currently loaded in memory. |
| model_version | str \| None | Loaded model version, if available. |
| uptime_seconds | float | Process uptime in seconds. |
