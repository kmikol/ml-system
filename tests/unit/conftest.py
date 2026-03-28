# tests/unit/conftest.py
"""
Set required environment variables before any service module is imported.

Several modules (serving/main.py, monitoring/ml_exporter/main.py, etc.) call
require_env() at module level, which calls sys.exit(1) if the variable is
absent.  pytest loads conftest.py before collecting test files, so setting
os.environ here guarantees the variables are visible when test modules import
those service modules.

DATA_CONTROLLER_DB_URL is intentionally omitted:
  ServingDataController gracefully degrades to a no-op when the URL is
  absent, which is the correct behaviour to exercise in unit tests.
"""

import os

os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("MODEL_STAGE", "Production")
os.environ.setdefault("SERVING_MODEL_POLL_INTERVAL", "60")
os.environ.setdefault("SERVING_SIMULATED_LATENCY_MS", "0")
# ModelArtifactController reads this at construction time (module-level in serving/main.py).
# A fake value is fine — tests patch load_from_mlflow before any real MLflow call is made.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

# training/main.py reads these at module level — required before any import of that module.
os.environ.setdefault("TRAINING_MAX_EPOCHS", "2")
os.environ.setdefault("TRAINING_SEED", "42")
os.environ.setdefault("TRAINING_BATCH_SIZE", "32")
os.environ.setdefault("TRAINING_LR", "0.001")

# monitoring/ml_exporter/main.py reads these at module level
os.environ.setdefault("DRIFT_POLL_INTERVAL", "5")
os.environ.setdefault("DRIFT_WINDOW_SECONDS", "300")
os.environ.setdefault("DRIFT_MIN_SAMPLES", "30")
