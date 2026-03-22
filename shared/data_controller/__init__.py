# shared/data_controller/__init__.py
"""
Data controller package — the single point of contact between the rest of the
system and the operational data storage backend (currently Postgres).

Architecture:
  _DataControllerBase      — connection lifecycle, table creation  (_base.py)
  ServingDataController    — store_prediction()                     (serving.py)
  DriftDataController      — get_predictions(), get_labeled_predictions() (drift.py)
  SamplingDataController   — get_predictions(), mark_candidate(), count_labels_since() (sampling.py)
  AnnotationDataController — write_label()                          (annotation.py)
  DatasetController        — versioned dataset samples in Postgres + MinIO (dataset.py)
  FakeDataController       — in-memory implementation for unit tests (fake.py)

Services import their specific controller directly, e.g.:
  from shared.data_controller.serving import ServingDataController
"""

from shared.data_controller._base import DataControllerError
from shared.data_controller.annotation import AnnotationDataController
from shared.data_controller.dataset import DatasetController
from shared.data_controller.drift import DriftDataController
from shared.data_controller.fake import FakeDataController
from shared.data_controller.sampling import SamplingDataController
from shared.data_controller.serving import ServingDataController

__all__ = [
    "DataControllerError",
    "ServingDataController",
    "DriftDataController",
    "SamplingDataController",
    "AnnotationDataController",
    "DatasetController",
    "FakeDataController",
]
