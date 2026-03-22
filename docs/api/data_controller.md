# Data Controller

Facade that hides **Postgres** and **MinIO S3** from the rest of the system. Services never import `psycopg2` or `boto3` directly — they instantiate the subclass appropriate for their role and call through its public methods.

All database errors are wrapped in `DataControllerError` so callers only need to handle one exception type.

## shared.data_controller

::: shared.data_controller
