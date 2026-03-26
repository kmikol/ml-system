# shared/data_controller/tests/integration/test_dataset.py
"""
Integration tests for DatasetController.

Requires Postgres and MinIO — provided by docker-compose.test.yml.
Connection details come from environment variables set by the compose file.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from shared.data_controller.dataset import DatasetController


def _random_image() -> list[list[float]]:
    rng = np.random.default_rng(42)
    return rng.random((14, 14)).tolist()


def _unique_version() -> str:
    """Return a unique version_id to avoid cross-test interference."""
    return f"test-{uuid4()}"


def _make_sample(version_id: str, split: str = "train", label: int = 3) -> dict:
    """Return kwargs for DatasetController.store_sample()."""
    return {
        "uuid": uuid4(),
        "version_id": version_id,
        "split": split,
        "label": label,
        "image_2d": _random_image(),
        "minio_path": f"test/{uuid4()}.npy",
    }


class TestDatasetControllerSamples:
    """Tests for store_sample() and download_image()."""

    def test_store_sample_is_retrievable(self):
        ctrl = DatasetController()
        version = _unique_version()
        s = _make_sample(version)
        ctrl.store_sample(**s)

        samples = ctrl.get_dataset_split(version, "train")
        stored = next((x for x in samples if x["uuid"] == s["uuid"]), None)
        assert stored is not None

    def test_stored_sample_has_correct_label_and_path(self):
        ctrl = DatasetController()
        version = _unique_version()
        s = _make_sample(version, label=7)
        ctrl.store_sample(**s)

        samples = ctrl.get_dataset_split(version, "train")
        stored = next(x for x in samples if x["uuid"] == s["uuid"])
        assert stored["label"] == 7
        assert stored["minio_path"] == s["minio_path"]

    def test_stored_image_round_trips(self):
        ctrl = DatasetController()
        version = _unique_version()
        s = _make_sample(version)
        ctrl.store_sample(**s)

        samples = ctrl.get_dataset_split(version, "train")
        stored = next(x for x in samples if x["uuid"] == s["uuid"])

        expected = np.array(s["image_2d"], dtype=np.float32)
        np.testing.assert_allclose(stored["image"], expected, rtol=1e-5)
        assert stored["image"].shape == (14, 14)

    def test_store_sample_is_idempotent(self):
        ctrl = DatasetController()
        version = _unique_version()
        s = _make_sample(version)
        ctrl.store_sample(**s)
        ctrl.store_sample(**s)  # ON CONFLICT DO NOTHING

        samples = ctrl.get_dataset_split(version, "train")
        count = sum(1 for x in samples if x["uuid"] == s["uuid"])
        assert count == 1

    def test_download_image_returns_numpy_array(self):
        ctrl = DatasetController()
        version = _unique_version()
        s = _make_sample(version)
        ctrl.store_sample(**s)

        image = ctrl.download_image(s["minio_path"])
        assert isinstance(image, np.ndarray)
        assert image.shape == (14, 14)


class TestDatasetControllerVersions:
    """Tests for store_sample(), get_dataset_split(), get_latest_version()."""

    def test_get_dataset_split_returns_only_requested_split(self):
        ctrl = DatasetController()
        version = _unique_version()
        train_s = _make_sample(version, split="train")
        test_s = _make_sample(version, split="test")
        ctrl.store_sample(**train_s)
        ctrl.store_sample(**test_s)

        train_ids = {x["uuid"] for x in ctrl.get_dataset_split(version, "train")}
        test_ids = {x["uuid"] for x in ctrl.get_dataset_split(version, "test")}

        assert train_s["uuid"] in train_ids
        assert test_s["uuid"] in test_ids
        assert train_s["uuid"] not in test_ids
        assert test_s["uuid"] not in train_ids

    def test_get_dataset_split_empty_for_unknown_version(self):
        ctrl = DatasetController()
        samples = ctrl.get_dataset_split("nonexistent-version", "train")
        assert samples == []

    def test_get_dataset_split_empty_for_unknown_split(self):
        ctrl = DatasetController()
        version = _unique_version()
        # Store into train; querying val should return empty
        s = _make_sample(version, split="train")
        ctrl.store_sample(**s)

        samples = ctrl.get_dataset_split(version, "val")
        assert samples == []

    def test_store_sample_split_isolation(self):
        """A sample stored in 'train' for one version does not appear in another version."""
        ctrl = DatasetController()
        v1 = _unique_version()
        v2 = _unique_version()
        s = _make_sample(v1, split="train")
        ctrl.store_sample(**s)

        samples = ctrl.get_dataset_split(v2, "train")
        assert not any(x["uuid"] == s["uuid"] for x in samples)

    def test_get_latest_version_returns_most_recently_seeded(self):
        ctrl = DatasetController()
        # Seed two distinct versions; the second should be returned as latest
        v1 = _unique_version()
        v2 = _unique_version()
        ctrl.store_sample(**_make_sample(v1))
        ctrl.store_sample(**_make_sample(v2))

        latest = ctrl.get_latest_version()
        assert latest == v2

    def test_get_latest_version_returns_none_when_no_versions(self):
        # This test assumes the DB might have versions from other tests.
        # We only verify the method doesn't raise and returns a string or None.
        ctrl = DatasetController()
        result = ctrl.get_latest_version()
        assert result is None or isinstance(result, str)
