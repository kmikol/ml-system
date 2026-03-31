# training/tests/unit/conftest.py
"""
Set required environment variables before any training module is imported.

training/main.py calls require_env() at module level, which calls sys.exit(1)
if the variable is absent.  pytest loads conftest.py before collecting test
files, so setting os.environ here guarantees the variables are visible when
test modules import the training module.
"""

import os

os.environ.setdefault("MODEL_NAME", "test-model")
# ModelArtifactController reads this at construction time (module-level in training/main.py).
# A fake value is fine — tests patch the controller before any real MLflow call is made.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

# training/main.py reads these at module level — required before any import of that module.
os.environ.setdefault("TRAINING_MAX_EPOCHS", "2")
os.environ.setdefault("TRAINING_SEED", "42")
os.environ.setdefault("TRAINING_BATCH_SIZE", "32")
os.environ.setdefault("TRAINING_LR", "0.001")
