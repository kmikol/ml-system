# tests/unit/test_sampling_job.py
"""
Unit tests for the sampling job (sampling/main.py).

All DB interactions are replaced by a MagicMock so no Postgres connection
is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from sampling.main import main


def _make_ctrl(marked: list | None = None) -> MagicMock:
    ctrl = MagicMock()
    ctrl.select_and_mark_candidates.return_value = marked or []
    return ctrl


class TestMainSelectsAndMarksCandidates:
    def test_calls_controller_with_configured_limit(self, monkeypatch):
        monkeypatch.setenv("SAMPLING_CANDIDATES_PER_RUN", "25")
        ctrl = _make_ctrl([uuid4()])

        with patch("sampling.main.SamplingDataController", return_value=ctrl):
            main()

        ctrl.select_and_mark_candidates.assert_called_once_with(limit=25, strategy="random")

    def test_uses_default_limit_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("SAMPLING_CANDIDATES_PER_RUN", raising=False)
        ctrl = _make_ctrl([uuid4()])

        with patch("sampling.main.SamplingDataController", return_value=ctrl):
            main()

        ctrl.select_and_mark_candidates.assert_called_once_with(limit=50, strategy="random")

    def test_uses_default_strategy_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("SAMPLING_STRATEGY", raising=False)
        ctrl = _make_ctrl([uuid4()])

        with patch("sampling.main.SamplingDataController", return_value=ctrl):
            main()

        ctrl.select_and_mark_candidates.assert_called_once_with(limit=50, strategy="random")

    def test_uses_low_confidence_strategy(self, monkeypatch):
        monkeypatch.setenv("SAMPLING_STRATEGY", "low_confidence")
        ctrl = _make_ctrl([uuid4()])

        with patch("sampling.main.SamplingDataController", return_value=ctrl):
            main()

        ctrl.select_and_mark_candidates.assert_called_once_with(limit=50, strategy="low_confidence")

    def test_uses_high_mahalanobis_strategy(self, monkeypatch):
        monkeypatch.setenv("SAMPLING_STRATEGY", "high_mahalanobis")
        ctrl = _make_ctrl([uuid4()])

        with patch("sampling.main.SamplingDataController", return_value=ctrl):
            main()

        ctrl.select_and_mark_candidates.assert_called_once_with(limit=50, strategy="high_mahalanobis")

    def test_uses_diverse_strategy(self, monkeypatch):
        monkeypatch.setenv("SAMPLING_STRATEGY", "diverse")
        ctrl = _make_ctrl([uuid4()])

        with patch("sampling.main.SamplingDataController", return_value=ctrl):
            main()

        ctrl.select_and_mark_candidates.assert_called_once_with(limit=50, strategy="diverse")

    def test_no_error_when_no_unannotated_predictions(self, monkeypatch):
        """Empty result from the DB is a normal operating condition."""
        monkeypatch.setenv("SAMPLING_CANDIDATES_PER_RUN", "50")
        ctrl = _make_ctrl([])  # nothing to mark

        with patch("sampling.main.SamplingDataController", return_value=ctrl):
            main()  # must not raise

    def test_marks_all_returned_uuids(self, monkeypatch):
        """The job forwards whatever the controller returns without filtering."""
        monkeypatch.setenv("SAMPLING_CANDIDATES_PER_RUN", "10")
        uuids = [uuid4() for _ in range(5)]
        ctrl = _make_ctrl(uuids)

        with patch("sampling.main.SamplingDataController", return_value=ctrl):
            main()  # result is logged, not returned — just verify no error

        ctrl.select_and_mark_candidates.assert_called_once_with(limit=10, strategy="random")


class TestMainEnvValidation:
    def test_exits_on_non_integer_candidates_per_run(self, monkeypatch):
        monkeypatch.setenv("SAMPLING_CANDIDATES_PER_RUN", "many")

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_exits_on_float_string(self, monkeypatch):
        monkeypatch.setenv("SAMPLING_CANDIDATES_PER_RUN", "3.5")

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_exits_on_invalid_strategy(self, monkeypatch):
        monkeypatch.setenv("SAMPLING_STRATEGY", "invalid_strategy")

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
