# tests/unit/test_annotation_job.py
"""
Unit tests for the annotation job (annotation/main.py).

All DB interactions are replaced by a MagicMock so no Postgres connection is
needed.  Oracle files are written to pytest tmp_path directories.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from annotation.main import _load_oracle, main

# ── _load_oracle ──────────────────────────────────────────────────────────────


class TestLoadOracle:
    def test_builds_uuid_to_label_mapping(self, tmp_path):
        uid1, uid2 = str(uuid4()), str(uuid4())
        np.save(tmp_path / "uuids.npy", np.array([uid1, uid2]))
        np.save(tmp_path / "labels.npy", np.array([3, 7]))

        oracle = _load_oracle(str(tmp_path))

        assert oracle == {uid1: 3, uid2: 7}

    def test_labels_are_integers(self, tmp_path):
        uid = str(uuid4())
        np.save(tmp_path / "uuids.npy", np.array([uid]))
        np.save(tmp_path / "labels.npy", np.array([5], dtype=np.int64))

        oracle = _load_oracle(str(tmp_path))

        assert isinstance(oracle[uid], int)

    def test_exits_when_uuids_npy_missing(self, tmp_path):
        np.save(tmp_path / "labels.npy", np.array([1]))
        # uuids.npy not created

        with pytest.raises(SystemExit) as exc:
            _load_oracle(str(tmp_path))
        assert exc.value.code == 1

    def test_exits_when_labels_npy_missing(self, tmp_path):
        np.save(tmp_path / "uuids.npy", np.array([str(uuid4())]))
        # labels.npy not created

        with pytest.raises(SystemExit) as exc:
            _load_oracle(str(tmp_path))
        assert exc.value.code == 1

    def test_exits_when_directory_missing(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(SystemExit) as exc:
            _load_oracle(str(nonexistent))
        assert exc.value.code == 1

    def test_empty_oracle(self, tmp_path):
        np.save(tmp_path / "uuids.npy", np.array([], dtype=str))
        np.save(tmp_path / "labels.npy", np.array([], dtype=int))

        oracle = _load_oracle(str(tmp_path))

        assert oracle == {}


# ── main() ────────────────────────────────────────────────────────────────────

# Helpers for building mock controllers used in main() tests.

def _make_ctrl(candidates: list, **kwargs) -> MagicMock:
    """Return a mock AnnotationDataController with given candidate UUIDs."""
    ctrl = MagicMock()
    ctrl.get_candidates.return_value = candidates
    for attr, val in kwargs.items():
        setattr(ctrl, attr, val)
    return ctrl


def _write_oracle(tmp_path, uuids: list[str], labels: list[int]) -> None:
    np.save(tmp_path / "uuids.npy", np.array(uuids))
    np.save(tmp_path / "labels.npy", np.array(labels))


class TestMainAnnotatesOracleHits:
    def test_write_label_called_for_known_uuid(self, tmp_path, monkeypatch):
        uid = uuid4()
        _write_oracle(tmp_path, [str(uid)], [5])
        monkeypatch.setenv("ANNOTATION_ORACLE_PATH", str(tmp_path))
        monkeypatch.setenv("ANNOTATION_SAMPLES_PER_RUN", "10")

        ctrl = _make_ctrl([uid])
        with patch("annotation.main.AnnotationDataController", return_value=ctrl):
            main()

        ctrl.write_label.assert_called_once_with(uid, 5)
        ctrl.reset_candidate.assert_not_called()

    def test_multiple_hits_all_labeled(self, tmp_path, monkeypatch):
        uids = [uuid4(), uuid4(), uuid4()]
        labels = [1, 2, 3]
        _write_oracle(tmp_path, [str(u) for u in uids], labels)
        monkeypatch.setenv("ANNOTATION_ORACLE_PATH", str(tmp_path))
        monkeypatch.setenv("ANNOTATION_SAMPLES_PER_RUN", "10")

        ctrl = _make_ctrl(uids)
        with patch("annotation.main.AnnotationDataController", return_value=ctrl):
            main()

        assert ctrl.write_label.call_count == 3
        ctrl.write_label.assert_any_call(uids[0], 1)
        ctrl.write_label.assert_any_call(uids[1], 2)
        ctrl.write_label.assert_any_call(uids[2], 3)


class TestMainOracleMiss:
    def test_reset_candidate_called_for_unknown_uuid(self, tmp_path, monkeypatch):
        """UUID not in oracle → reset_candidate, not write_label."""
        uid_known = uuid4()
        uid_unknown = uuid4()
        _write_oracle(tmp_path, [str(uid_known)], [4])
        monkeypatch.setenv("ANNOTATION_ORACLE_PATH", str(tmp_path))
        monkeypatch.setenv("ANNOTATION_SAMPLES_PER_RUN", "10")

        ctrl = _make_ctrl([uid_unknown])
        with patch("annotation.main.AnnotationDataController", return_value=ctrl):
            main()

        ctrl.reset_candidate.assert_called_once_with(uid_unknown)
        ctrl.write_label.assert_not_called()

    def test_mixed_hit_and_miss(self, tmp_path, monkeypatch):
        uid_hit = uuid4()
        uid_miss = uuid4()
        _write_oracle(tmp_path, [str(uid_hit)], [9])
        monkeypatch.setenv("ANNOTATION_ORACLE_PATH", str(tmp_path))
        monkeypatch.setenv("ANNOTATION_SAMPLES_PER_RUN", "10")

        ctrl = _make_ctrl([uid_hit, uid_miss])
        with patch("annotation.main.AnnotationDataController", return_value=ctrl):
            main()

        ctrl.write_label.assert_called_once_with(uid_hit, 9)
        ctrl.reset_candidate.assert_called_once_with(uid_miss)


class TestMainEmptyCandidates:
    def test_no_writes_when_no_candidates(self, tmp_path, monkeypatch):
        _write_oracle(tmp_path, [str(uuid4())], [0])
        monkeypatch.setenv("ANNOTATION_ORACLE_PATH", str(tmp_path))
        monkeypatch.setenv("ANNOTATION_SAMPLES_PER_RUN", "10")

        ctrl = _make_ctrl([])
        with patch("annotation.main.AnnotationDataController", return_value=ctrl):
            main()  # must not raise

        ctrl.write_label.assert_not_called()
        ctrl.reset_candidate.assert_not_called()


class TestMainEnvValidation:
    def test_exits_on_non_integer_samples_per_run(self, monkeypatch):
        monkeypatch.setenv("ANNOTATION_SAMPLES_PER_RUN", "ten")
        monkeypatch.setenv("ANNOTATION_ORACLE_PATH", "/unused")

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_uses_default_samples_per_run_when_unset(self, tmp_path, monkeypatch):
        """When ANNOTATION_SAMPLES_PER_RUN is absent, default of 10 is used."""
        uid = uuid4()
        _write_oracle(tmp_path, [str(uid)], [2])
        monkeypatch.setenv("ANNOTATION_ORACLE_PATH", str(tmp_path))
        monkeypatch.delenv("ANNOTATION_SAMPLES_PER_RUN", raising=False)

        ctrl = _make_ctrl([uid])
        with patch("annotation.main.AnnotationDataController", return_value=ctrl):
            main()

        ctrl.get_candidates.assert_called_once_with(limit=10)
