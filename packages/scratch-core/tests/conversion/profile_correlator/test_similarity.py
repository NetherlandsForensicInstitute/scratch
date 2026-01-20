"""Tests for the similarity module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from conversion.profile_correlator import (
    compute_cross_correlation,
    compute_comparison_metrics,
    TransformParameters,
)


class TestComputeCrossCorrelation:
    """Tests for compute_cross_correlation function."""

    def test_identical_profiles_give_correlation_one(self):
        """Identical profiles should have correlation coefficient of 1.0."""
        profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_cross_correlation(profile, profile)
        assert_allclose(result, 1.0, atol=1e-10)

    def test_negatively_correlated_profiles(self):
        """Negatively correlated profiles should have correlation near -1.0."""
        p1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = compute_cross_correlation(p1, p2)
        assert_allclose(result, -1.0, atol=1e-10)

    def test_uncorrelated_profiles(self):
        """Uncorrelated profiles should have correlation near 0."""
        np.random.seed(42)
        p1 = np.random.randn(1000)
        p2 = np.random.randn(1000)
        result = compute_cross_correlation(p1, p2)
        assert abs(result) < 0.1  # Should be close to 0 for random data

    def test_handles_nan_values(self):
        """NaN values should be excluded from correlation calculation."""
        p1 = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        p2 = np.array([1.1, 2.0, 2.9, 4.0, 5.1])
        result = compute_cross_correlation(p1, p2)
        # Should still compute valid correlation using non-NaN values
        assert not np.isnan(result)
        assert result > 0.9  # Should be highly correlated

    def test_all_nan_returns_nan(self):
        """If all values are NaN, should return NaN."""
        p1 = np.array([np.nan, np.nan, np.nan])
        p2 = np.array([1.0, 2.0, 3.0])
        result = compute_cross_correlation(p1, p2)
        assert np.isnan(result)

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
        assert np.isnan(result)

    def test_sinusoidal_profiles(self):
        """Test correlation of sinusoidal profiles."""
        x = np.linspace(0, 2 * np.pi, 100)
        p1 = np.sin(x)
        p2 = np.sin(x + 0.1)  # Slightly phase-shifted
        result = compute_cross_correlation(p1, p2)
        assert result > 0.95  # Should be very highly correlated


class TestComputeComparisonMetrics:
    """Tests for compute_comparison_metrics function."""

    def test_returns_comparison_results_object(self):
        """Should return a ComparisonResults dataclass."""
        transforms = [TransformParameters(translation=0.0, scaling=1.0)]
        p1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-6  # Heights in meters
        p2 = np.array([1.1, 2.1, 2.9, 4.0, 5.1]) * 1e-6
        result = compute_comparison_metrics(transforms, p1, p2, pixel_size_um=0.5)

        assert result.is_profile_comparison is True
        assert not np.isnan(result.correlation_coefficient)

    def test_position_shift_with_translation(self):
        """Position shift should reflect translation in micrometers."""
        transforms = [TransformParameters(translation=10.0, scaling=1.0)]
        p1 = np.zeros(100)
        p2 = np.zeros(100)
        pixel_size_um = 0.5
        result = compute_comparison_metrics(transforms, p1, p2, pixel_size_um)

        # Position shift = translation * pixel_size = 10 * 0.5 = 5 um
        assert_allclose(result.position_shift, 5.0, atol=1e-10)

    def test_scale_factor_from_transforms(self):
        """Scale factor should be accumulated from transforms."""
        transforms = [
            TransformParameters(translation=0.0, scaling=1.01),
            TransformParameters(translation=0.0, scaling=1.02),
        ]
        p1 = np.zeros(100)
        p2 = np.zeros(100)
        result = compute_comparison_metrics(transforms, p1, p2, pixel_size_um=0.5)

        # Total scaling = 1.01 * 1.02 = 1.0302
        assert_allclose(result.scale_factor, 1.0302, atol=1e-10)

    def test_roughness_metrics_computed(self):
        """Sa and Sq roughness metrics should be computed."""
        transforms = [TransformParameters(translation=0.0, scaling=1.0)]
        p1 = np.array([1.0, -1.0, 1.0, -1.0, 1.0]) * 1e-6
        p2 = np.array([1.5, -1.5, 1.5, -1.5, 1.5]) * 1e-6
        result = compute_comparison_metrics(transforms, p1, p2, pixel_size_um=0.5)

        # Sa = mean(|profile|), converted to um
        assert result.sa_ref > 0
        assert result.sa_comp > 0
        assert result.sq_ref > 0
        assert result.sq_comp > 0

    def test_overlap_length_computed(self):
        """Overlap length should be computed correctly."""
        transforms = [TransformParameters(translation=0.0, scaling=1.0)]
        n_samples = 100
        p1 = np.zeros(n_samples)
        p2 = np.zeros(n_samples)
        pixel_size_um = 0.5
        result = compute_comparison_metrics(transforms, p1, p2, pixel_size_um)

        # Overlap length = n_samples * pixel_size = 100 * 0.5 = 50 um
        assert_allclose(result.overlap_length, 50.0, atol=1e-10)
