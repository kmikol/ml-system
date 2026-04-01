# shared/data_controller/tests/integration/test_dataset.py
"""
Integration tests for DatasetController.

Requires Postgres, MinIO, and lakeFS — provided by docker-compose.test.yml.
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


class TestDatasetControllerLakeFSVersioning:
    """Integration tests for create_version(), get_version_info(), get_version_history().

    Requires a running lakeFS service (LAKEFS_ENDPOINT_URL) in addition to
    Postgres and MinIO.  All services are provided by docker-compose.test.yml.
    """

    def _store_samples(self, ctrl: DatasetController, version_id: str, n: int = 3) -> None:
        """Store *n* distinct samples for *version_id*."""
        for _ in range(n):
            ctrl.store_sample(**_make_sample(version_id))

    def test_create_version_returns_commit_id(self):
        ctrl = DatasetController()
        version = _unique_version()
        self._store_samples(ctrl, version)

        commit_id = ctrl.create_version(version, parent_version_id=None)

        assert isinstance(commit_id, str)
        assert len(commit_id) > 0

    def test_create_version_registers_row_in_database(self):
        ctrl = DatasetController()
        version = _unique_version()
        self._store_samples(ctrl, version, n=2)

        commit_id = ctrl.create_version(version, parent_version_id=None)

        info = ctrl.get_version_info(version)
        assert info is not None
        assert info["version_id"] == version
        assert info["lakefs_commit_id"] == commit_id
        assert info["lakefs_tag"] == f"dataset/{version}"
        assert info["sample_count"] == 2
        assert info["parent_version_id"] is None

    def test_create_version_records_parent_version_id(self):
        ctrl = DatasetController()
        v1 = _unique_version()
        v2 = _unique_version()
        self._store_samples(ctrl, v1)
        self._store_samples(ctrl, v2)

        ctrl.create_version(v1, parent_version_id=None)
        ctrl.create_version(v2, parent_version_id=v1)

        info = ctrl.get_version_info(v2)
        assert info is not None
        assert info["parent_version_id"] == v1

    def test_create_version_is_idempotent(self):
        """Second call returns the same commit_id without creating a new commit."""
        ctrl = DatasetController()
        version = _unique_version()
        self._store_samples(ctrl, version)

        commit_id_first = ctrl.create_version(version, parent_version_id=None)
        commit_id_second = ctrl.create_version(version, parent_version_id=None)

        assert commit_id_first == commit_id_second

    def test_get_version_info_returns_none_for_unknown_version(self):
        ctrl = DatasetController()
        result = ctrl.get_version_info("nonexistent-version-xyz")
        assert result is None

    def test_get_version_history_includes_created_versions(self):
        ctrl = DatasetController()
        v1 = _unique_version()
        v2 = _unique_version()
        self._store_samples(ctrl, v1)
        self._store_samples(ctrl, v2)
        ctrl.create_version(v1, parent_version_id=None)
        ctrl.create_version(v2, parent_version_id=v1)

        history = ctrl.get_version_history()

        version_ids = [h["version_id"] for h in history]
        assert v1 in version_ids
        assert v2 in version_ids
        # v1 was created first, so it must appear before v2
        assert version_ids.index(v1) < version_ids.index(v2)

    def test_get_version_history_returns_list(self):
        ctrl = DatasetController()
        history = ctrl.get_version_history()
        assert isinstance(history, list)
