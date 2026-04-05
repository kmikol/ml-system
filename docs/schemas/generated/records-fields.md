## PredictRecord

### Fields

| Field | Type | Description |
|------|------|-------------|
| uuid | UUID | Unique prediction record identifier. |
| timestamp | datetime | UTC timestamp when prediction was produced. |
| model_version | str | Model version that generated this prediction. |
| embedding | list[float] | Embedding vector extracted from serving model. |
| prediction | int | Predicted class index. |
| confidence | float | Confidence score of predicted class. |
| prediction_distribution | list[float] | Full class-probability distribution returned by model. |
| annotation_status | Literal['none', 'candidate', 'annotated'] | Annotation workflow state for this sample. |
| annotated_label | int \| None | Human-provided label written by annotation job when available. |

## InferenceEvent

### Fields

| Field | Type | Description |
|------|------|-------------|
| event_id | str | Unique event identifier. |
| timestamp | datetime | UTC timestamp of inference event. |
| model_version | str | Model version used for this inference. |
| request_id | str | Request correlation identifier. |
| image | list[list[float]] | Input 14x14 grayscale image. |
| embedding | list[float] | Model embedding vector for the sample. |
| prediction | int | Predicted class index. |
| confidence | float | Confidence score for predicted class. |
| prediction_distribution | list[float] | Full class-probability distribution. |
| schema_version | str | Event payload schema version. |

## Feature Constants

| Constant | Value | Description |
|----------|-------|-------------|
| IMAGE_SIZE | (14, 14) | Input image dimensions used by the model pipeline. |
| INPUT_DIM | 14 * 14 | Flattened input size derived from IMAGE_SIZE. |
| NUM_CLASSES | 10 | Number of output digit classes. |
| EMBEDDING_DIM | 64 | Embedding vector dimension used in model outputs. |
