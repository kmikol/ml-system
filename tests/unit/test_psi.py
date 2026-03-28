# tests/unit/test_psi.py
"""
Unit tests for compute_psi() in monitoring/ml_exporter/main.py.

The redesigned module has no module-level side effects, so compute_psi can
be imported directly without any mocking or workarounds.
"""

import math

import pytest

from monitoring.ml_exporter.main import compute_psi


class TestPsiIdenticalDistributions:
    def test_uniform_identical(self):
        """Identical distributions → PSI is 0."""
        dist = [0.1] * 10
        assert compute_psi(dist, dist) == pytest.approx(0.0, abs=1e-9)

    def test_non_uniform_identical(self):
        """Non-uniform but identical → still 0."""
        dist = [0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1]
        assert compute_psi(dist, dist) == pytest.approx(0.0, abs=1e-9)


class TestPsiPositiveForShiftedDistributions:
    def test_positive_for_any_shift(self):
        """Any distribution shift produces PSI > 0."""
        uniform = [0.1] * 10
        shifted = [0.91] + [0.01] * 9
        assert compute_psi(shifted, uniform) > 0.0

    def test_exceeds_threshold_for_large_shift(self):
        """Large shift (all predictions on one class) exceeds PSI 0.25 threshold."""
        uniform = [0.1] * 10
        all_on_one = [1.0] + [0.0] * 9
        assert compute_psi(all_on_one, uniform) > 0.25

    def test_below_threshold_for_small_shift(self):
        """Small shift stays below 0.10."""
        reference = [0.1] * 10
        actual = [0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08]
        assert compute_psi(actual, reference) < 0.10

    def test_symmetric_shift_is_nonzero(self):
        """Moving weight between classes in opposite directions is detected."""
        reference = [0.1] * 10
        actual = [0.3, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1]
        assert compute_psi(actual, reference) > 0.0


class TestPsiNumericalStability:
    def test_zero_probability_class_no_error(self):
        """A class with probability 0 does not raise a log(0) error."""
        reference = [0.1] * 10
        actual = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
        psi = compute_psi(actual, reference)
        assert math.isfinite(psi)
        assert psi > 0.0

    def test_both_zero_same_class_returns_finite(self):
        """When both actual and reference are 0 for a class, the term is ~0."""
        reference = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
        actual = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert compute_psi(actual, reference) == pytest.approx(0.0, abs=1e-9)
