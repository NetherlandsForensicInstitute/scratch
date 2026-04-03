import numpy as np
from numpy.testing import assert_allclose

from conversion.profile_correlator.profile_correlator import _correlations_at_all_shifts


class TestCorrelationsAtAllShifts:
    """Tests for _correlations_at_all_shifts function."""

    def test_identical_profiles_peak_at_zero_shift(self):
        """Identical profiles should have correlation 1.0 at shift 0."""
        profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        correlations = _correlations_at_all_shifts(
            profile, profile, min_overlap_samples=2
        )
        # Shift 0 corresponds to index len(profile) - 1
        zero_shift_idx = len(profile) - 1
        assert_allclose(correlations[zero_shift_idx], 1.0, atol=1e-10)

    def test_best_shift_is_zero_for_identical_profiles(self):
        """Best correlation should be at shift 0 for identical profiles."""
        profile = np.random.default_rng(42).normal(size=100)
        correlations = _correlations_at_all_shifts(
            profile, profile, min_overlap_samples=20
        )
        best_idx = np.argmax(correlations)
        best_shift = best_idx - (len(profile) - 1)
        assert best_shift == 0

    def test_negatively_correlated_profiles(self):
        """Reversed profile should have correlation -1.0 at shift 0."""
        p1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        correlations = _correlations_at_all_shifts(p1, p2, min_overlap_samples=2)
        zero_shift_idx = len(p2) - 1
        assert_allclose(correlations[zero_shift_idx], -1.0, atol=1e-10)

    def test_known_shift_is_found(self):
        """A shifted copy should have peak correlation at the correct shift."""
        rng = np.random.default_rng(42)
        full_signal = rng.normal(size=50)
        shift = 10
        reference = full_signal
        compared = full_signal[shift : shift + 20]
        correlations = _correlations_at_all_shifts(
            reference, compared, min_overlap_samples=10
        )
        best_idx = np.argmax(correlations)
        best_shift = best_idx - (len(compared) - 1)
        assert best_shift == shift

    def test_handles_nan_values(self):
        """NaN values should not prevent finding the correct correlation."""
        p1 = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        p2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        correlations = _correlations_at_all_shifts(p1, p2, min_overlap_samples=2)
        zero_shift_idx = len(p2) - 1
        assert correlations[zero_shift_idx] > 0.9

    def test_all_nan_returns_neg_inf(self):
        """All-NaN profile should give -inf at every shift."""
        p1 = np.array([np.nan, np.nan, np.nan])
        p2 = np.array([1.0, 2.0, 3.0])
        correlations = _correlations_at_all_shifts(p1, p2, min_overlap_samples=2)
        assert np.all(np.isneginf(correlations))

    def test_constant_profile_returns_neg_inf(self):
        """Constant profile (zero variance) should give -inf."""
        p1 = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        p2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        correlations = _correlations_at_all_shifts(p1, p2, min_overlap_samples=2)
        zero_shift_idx = len(p2) - 1
        assert np.isneginf(correlations[zero_shift_idx])

    def test_insufficient_overlap_returns_neg_inf(self):
        """Shifts with too few overlapping samples should give -inf."""
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])
        correlations = _correlations_at_all_shifts(p1, p2, min_overlap_samples=3)
        # At extreme shifts, overlap < 3 → should be -inf
        assert np.isneginf(correlations[0])
        assert np.isneginf(correlations[-1])

    def test_sinusoidal_phase_shift_detected(self):
        """A phase-shifted sine should have peak correlation at the correct offset."""
        x = np.linspace(0, 4 * np.pi, 200)
        reference = np.sin(x)
        shift = 10
        compared = np.sin(x[shift : shift + 100])
        correlations = _correlations_at_all_shifts(
            reference, compared, min_overlap_samples=20
        )
        best_idx = np.argmax(correlations)
        best_shift = best_idx - (len(compared) - 1)
        assert abs(best_shift - shift) <= 1
