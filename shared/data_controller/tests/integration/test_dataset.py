# shared/data_controller/tests/integration/test_dataset.py
"""
Integration tests for DatasetController.

Requires Postgres and MinIO — provided by docker-compose.test.yml.
Connection details come from environment variables set by the compose file.
"""

from __future__ import annotations

import uuid

import numpy as np

from shared.data_controller.dataset import DatasetController


def _random_image() -> list[list[float]]:
    rng = np.random.default_rng(42)
    return rng.random((14, 14)).tolist()


def _make_sample(split: str = "train", label: int = 3) -> dict:
    return {
        "sample_id": str(uuid.uuid4()),
        "split": split,
        "label": label,
        "image_2d": _random_image(),
        "minio_path": f"test/{uuid.uuid4()}.npy",
    }


class TestDatasetController:
    def test_store_sample_is_retrievable(self):
        ctrl = DatasetController()
        s = _make_sample()
        ctrl.store_sample(**s)

        samples = ctrl.get_dataset_split(s["split"])
        stored = next((x for x in samples if x["sample_id"] == s["sample_id"]), None)
        assert stored is not None

    def test_stored_sample_has_correct_label_and_path(self):
        ctrl = DatasetController()
        s = _make_sample(split="val", label=7)
        ctrl.store_sample(**s)

        samples = ctrl.get_dataset_split("val")
        stored = next(x for x in samples if x["sample_id"] == s["sample_id"])
        assert stored["label"] == 7
        assert stored["minio_path"] == s["minio_path"]

    def test_stored_image_round_trips(self):
        ctrl = DatasetController()
        s = _make_sample()
        ctrl.store_sample(**s)

        samples = ctrl.get_dataset_split(s["split"])
        stored = next(x for x in samples if x["sample_id"] == s["sample_id"])

        expected = np.array(s["image_2d"], dtype=np.float32)
        np.testing.assert_allclose(stored["image"], expected, rtol=1e-5)
        assert stored["image"].shape == (14, 14)

    def test_store_is_idempotent(self):
        ctrl = DatasetController()
        s = _make_sample()
        ctrl.store_sample(**s)
        ctrl.store_sample(**s)  # ON CONFLICT DO NOTHING

        samples = ctrl.get_dataset_split(s["split"])
        count = sum(1 for x in samples if x["sample_id"] == s["sample_id"])
        assert count == 1

    def test_get_dataset_split_returns_only_requested_split(self):
        ctrl = DatasetController()
        train_sample = _make_sample(split="train")
        test_sample = _make_sample(split="test")
        ctrl.store_sample(**train_sample)
        ctrl.store_sample(**test_sample)

        train_ids = {x["sample_id"] for x in ctrl.get_dataset_split("train")}
        test_ids = {x["sample_id"] for x in ctrl.get_dataset_split("test")}

        assert train_sample["sample_id"] in train_ids
        assert test_sample["sample_id"] in test_ids
        assert train_sample["sample_id"] not in test_ids
        assert test_sample["sample_id"] not in train_ids

    def test_download_image_returns_numpy_array(self):
        ctrl = DatasetController()
        s = _make_sample()
        ctrl.store_sample(**s)

        image = ctrl.download_image(s["minio_path"])
        assert isinstance(image, np.ndarray)
        assert image.shape == (14, 14)

    def test_get_dataset_split_empty_for_unknown_split(self):
        ctrl = DatasetController()
        samples = ctrl.get_dataset_split("nonexistent_split")
        assert samples == []
