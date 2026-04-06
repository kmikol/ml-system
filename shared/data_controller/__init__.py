# shared/data_controller/__init__.py
"""
Data controller package — the single point of contact between the rest of the
system and the operational data storage backend (Postgres + object storage + lakeFS).

Architecture:

    _DataControllerBase      — connection lifecycle, schema creation (_base.py)
    ServingDataController    — store_prediction()                    (serving.py)
    DriftDataController      — get_predictions(), get_annotated_count(), get_labeled_predictions() (drift.py)
    SamplingDataController   — select_and_mark_candidates()          (sampling.py)
    AnnotationDataController — get_candidates(), write_label()       (annotation.py)
    DatasetController        — samples, dataset versions, splits in Postgres + object storage + lakeFS (dataset.py)
    FakeDataController       — in-memory implementation for unit tests (fake.py)

    ObjectStore / MinIOObjectStore — swappable array storage backend (_object_store.py)
    LakeFSClient                  — lakeFS SDK wrapper (_lakefs.py)

Services import their specific controller directly:

    from shared.data_controller.serving import ServingDataController
"""

from shared.data_controller._base import DataControllerError

__all__ = [
    "DataControllerError",
]
