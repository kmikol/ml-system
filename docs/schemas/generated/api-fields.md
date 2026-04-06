## PredictRequest

### Parameters

- **image** (list[list[float]]; required): 14x14 grayscale image, values in [0, 1]
- **uuid** (UUID | None; optional; default=None): UUID of this sample, if known (enables annotation pipeline)

## PredictResponse

### Parameters

- **prediction** (int; required): Predicted class index.
- **confidence** (float; required): Confidence score for predicted class.
- **model_version** (str; required): Model version used to generate prediction.
- **uuid** (UUID; required): UUID associated with this prediction record.

## ValidationErrorResponse

### Parameters

- **detail** (str; required): High-level error summary.
- **errors** (list[dict]; required): Detailed validation errors.

## HealthResponse

### Parameters

- **status** (str; required): Service health state, e.g. healthy/unhealthy.
- **model_loaded** (bool; required): Whether model is currently loaded in memory.
- **model_version** (str | None; optional; default=None): Loaded model version, if available.
- **uptime_seconds** (float; required): Process uptime in seconds.
