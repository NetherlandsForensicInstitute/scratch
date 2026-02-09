"""Tests for the similarity module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from conversion.profile_correlator import (
    compute_cross_correlation,
)


class TestComputeCrossCorrelation:
    """Tests for compute_cross_correlation function."""

    def test_identical_profiles_give_correlation_one(self):
        """Identical profiles should have correlation coefficient of 1.0."""
        profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_cross_correlation(profile, profile)
        assert result is not None
        assert_allclose(result, 1.0, atol=1e-10)

    def test_negatively_correlated_profiles(self):
        """Negatively correlated profiles should have correlation near -1.0."""
        p1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = compute_cross_correlation(p1, p2)
        assert result is not None
        assert_allclose(result, -1.0, atol=1e-10)

    def test_uncorrelated_profiles(self):
        """Uncorrelated profiles should have correlation near 0."""
        np.random.seed(42)
        p1 = np.random.randn(1000)
        p2 = np.random.randn(1000)
        result = compute_cross_correlation(p1, p2)
        assert result is not None
        assert abs(result) < 0.1  # Should be close to 0 for random data

    def test_handles_nan_values(self):
        """NaN values should be excluded from correlation calculation."""
        p1 = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        p2 = np.array([1.1, 2.0, 2.9, 4.0, 5.1])
        result = compute_cross_correlation(p1, p2)
        assert result is not None
        # Should still compute valid correlation using non-NaN values
        assert not np.isnan(result)
        assert result > 0.9  # Should be highly correlated

    def test_all_nan_returns_none(self):
        """If all values are NaN, should return None."""
        p1 = np.array([np.nan, np.nan, np.nan])
        p2 = np.array([1.0, 2.0, 3.0])
        result = compute_cross_correlation(p1, p2)
        assert result is None

    def test_different_lengths_raises_error(self):
        """Profiles with different lengths should raise ValueError."""
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            compute_cross_correlation(p1, p2)

    def test_constant_profile_returns_nan(self):
        """Constant profile (zero variance) should return NaN."""
        p1 = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        p2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_cross_correlation(p1, p2)
        assert result is not None
        assert np.isnan(result)

    def test_sinusoidal_profiles(self):
        """Test correlation of sinusoidal profiles."""
        x = np.linspace(0, 2 * np.pi, 100)
        p1 = np.sin(x)
        p2 = np.sin(x + 0.1)  # Slightly phase-shifted
        result = compute_cross_correlation(p1, p2)
        assert result is not None
        assert result > 0.95  # Should be very highly correlated
