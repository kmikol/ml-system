# shared/data_controller/tests/integration/conftest.py
"""
Integration test fixtures for data_controller.

Infrastructure (Postgres, MinIO) is provided by docker-compose.test.yml.
Connection details come from environment variables set by the compose file,
matching the Kubernetes service topology:

  DATA_CONTROLLER_DB_URL       — Postgres  (service: postgres)
  DATASET_S3_ENDPOINT_URL      — MinIO API (service: minio)
  DATASET_BUCKET               — S3 bucket name
  AWS_ACCESS_KEY_ID / SECRET   — MinIO credentials

All controllers read these variables on construction — no fixture wiring needed.
"""
