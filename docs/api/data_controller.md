# Data Controller

Facade that hides **Postgres** and **MinIO S3** from the rest of the system. Services never import `psycopg2` or `boto3` directly — they instantiate the subclass appropriate for their role and call through its public methods.

All database errors are wrapped in `DataControllerError` so callers only need to handle one exception type.

Scope boundary:
- Owns operational data only (prediction records, labels, dataset samples).
- Does **not** own MLflow model artifact paths, model bundle resolution, or model registry lookups.
- Callers must use the Model Artifact Controller for model-related retrieval and publication.

## shared.data_controller

::: shared.data_controller
