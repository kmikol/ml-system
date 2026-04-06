## PredictRecord

### Parameters

- **uuid** (UUID; required): Unique prediction record identifier.
- **timestamp** (datetime; required): UTC timestamp when prediction was produced.
- **model_version** (str; required): Model version that generated this prediction.
- **embedding** (list[float]; required): Embedding vector extracted from serving model.
- **prediction** (int; required): Predicted class index.
- **confidence** (float; required): Confidence score of predicted class.
- **prediction_distribution** (list[float]; required): Full class-probability distribution returned by model.
- **annotation_status** (Literal['none', 'candidate', 'annotated']; optional; default='none'): Annotation workflow state for this sample.
- **annotated_label** (int | None; optional; default=None): Human-provided label written by annotation job when available.

## InferenceEvent

### Parameters

- **event_id** (str; required): Unique event identifier.
- **timestamp** (datetime; required): UTC timestamp of inference event.
- **model_version** (str; required): Model version used for this inference.
- **request_id** (str; required): Request correlation identifier.
- **image** (list[list[float]]; required): Input 14x14 grayscale image.
- **embedding** (list[float]; required): Model embedding vector for the sample.
- **prediction** (int; required): Predicted class index.
- **confidence** (float; required): Confidence score for predicted class.
- **prediction_distribution** (list[float]; required): Full class-probability distribution.
- **schema_version** (str; optional; default='1.0'): Event payload schema version.

## Feature Constants

### Attributes

- **IMAGE_SIZE** ((14, 14)): Input image dimensions used by the model pipeline.
- **INPUT_DIM** (14 * 14): Flattened input size derived from IMAGE_SIZE.
- **NUM_CLASSES** (10): Number of output digit classes.
- **EMBEDDING_DIM** (64): Embedding vector dimension used in model outputs.
