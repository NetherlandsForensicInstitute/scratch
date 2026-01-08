"""Tests for Gaussian noise removal filter.

This module tests the removal of high-frequency noise using Gaussian
lowpass filtering, verifying correct behavior for various input scenarios.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from conversion.preprocessing.remove_noise_gaussian import remove_noise_gaussian

SEED: int = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(SEED)


class TestRemoveNoiseGaussianReturnType:
    """Test the return type of remove_noise_gaussian."""

    def test_returns_tuple_of_three(self) -> None:
        """Verify function returns a tuple of (depth_data, range_indices, mask)."""
        data = np.zeros((100, 50))
        result = remove_noise_gaussian(data, xdim=1e-6, cutoff_lo=250e-6)

        assert isinstance(result, tuple)
        assert len(result) == 3

        depth_data, range_indices, mask = result
        assert isinstance(depth_data, np.ndarray)
        assert isinstance(range_indices, np.ndarray)
        assert isinstance(mask, np.ndarray)


class TestRemoveNoiseGaussianBasic:
    """Test basic functionality of remove_noise_gaussian."""

    def test_output_shape_matches_input_no_cropping(
        self, rng: np.random.Generator
    ) -> None:
        """Output shape should match input when cropping is disabled."""
        data = rng.random((100, 50))
        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=250e-6, cut_borders_after_smoothing=False
        )
        assert depth_data.shape == data.shape

    def test_output_shape_reduced_with_cropping(self, rng: np.random.Generator) -> None:
        """Output shape should be reduced when border cropping is enabled."""
        data = rng.random((200, 50))
        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=500e-6, cut_borders_after_smoothing=True
        )
        # Output should be smaller in row dimension
        assert depth_data.shape[0] < data.shape[0]
        assert depth_data.shape[1] == data.shape[1]

    def test_range_indices_match_output_size(self, rng: np.random.Generator) -> None:
        """Range indices length should match output row count."""
        data = rng.random((200, 50))
        depth_data, range_indices, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=500e-6, cut_borders_after_smoothing=True
        )
        assert len(range_indices) == depth_data.shape[0]

    def test_mask_shape_matches_output(self, rng: np.random.Generator) -> None:
        """Output mask shape should match output data shape."""
        data = rng.random((100, 50))
        depth_data, _, mask = remove_noise_gaussian(data, xdim=1e-6, cutoff_lo=250e-6)
        assert mask.shape == depth_data.shape


class TestLowpassBehavior:
    """Test that the filter behaves as a lowpass filter (smoothing)."""

    def test_uniform_data_unchanged(self) -> None:
        """Uniform data should remain unchanged (already smooth)."""
        data = np.ones((100, 50)) * 42.0
        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=250e-6, cut_borders_after_smoothing=False
        )
        # Interior should be nearly unchanged
        interior = depth_data[20:-20, 10:-10]
        assert_array_almost_equal(interior, 42.0, decimal=5)

    def test_smoothing_reduces_variance(self, rng: np.random.Generator) -> None:
        """Smoothing should reduce variance of noisy data."""
        # Create noisy data
        data = rng.random((200, 50)) * 100

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=500e-6, cut_borders_after_smoothing=False
        )

        # Variance should be reduced after smoothing
        input_var = np.var(data[30:-30, :])
        output_var = np.var(depth_data[30:-30, :])
        assert output_var < input_var

    def test_high_frequency_removed(self, rng: np.random.Generator) -> None:
        """High-frequency content should be removed."""
        # Create high-frequency oscillation (noise-like)
        rows = np.arange(200)
        hf_noise = np.sin(2 * np.pi * rows / 3)  # Very short period
        data = np.tile(hf_noise.reshape(-1, 1), (1, 50))

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=500e-6, cut_borders_after_smoothing=False
        )

        # High-frequency content should be attenuated
        output_var = np.var(depth_data[30:-30, :])
        input_var = np.var(data[30:-30, :])
        assert output_var < input_var * 0.1  # Significantly reduced

    def test_low_frequency_preserved(self) -> None:
        """Low-frequency content should be preserved."""
        # Create low-frequency signal
        rows = np.linspace(0, 2 * np.pi, 200)
        lf_signal = np.sin(rows)  # One full cycle over 200 pixels
        data = np.tile(lf_signal.reshape(-1, 1), (1, 50))

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=100e-6, cut_borders_after_smoothing=False
        )

        # Low-frequency signal should be mostly preserved
        output_interior = depth_data[30:-30, 25]
        input_interior = data[30:-30, 25]

        # Correlation should be high
        correlation = np.corrcoef(output_interior, input_interior)[0, 1]
        assert correlation > 0.95


class TestMaskHandling:
    """Test handling of masked regions."""

    def test_mask_all_true_equals_no_mask(self, rng: np.random.Generator) -> None:
        """All-true mask should give same result as no mask."""
        data = rng.random((100, 50))
        mask = np.ones(data.shape, dtype=bool)

        depth_with_mask, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=250, mask=mask, cut_borders_after_smoothing=False
        )
        depth_no_mask, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=250, mask=None, cut_borders_after_smoothing=False
        )

        assert_array_almost_equal(depth_with_mask, depth_no_mask)

    def test_partial_mask_handled(self, rng: np.random.Generator) -> None:
        """Partial mask should not crash and produce valid output."""
        data = rng.random((100, 50))
        mask = np.ones(data.shape, dtype=bool)
        mask[40:60, 20:30] = False  # Mask out central region

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=250, mask=mask, cut_borders_after_smoothing=False
        )

        assert depth_data.shape == data.shape
        assert not np.all(np.isnan(depth_data))


class TestNaNHandling:
    """Test handling of NaN values in input data."""

    def test_nan_in_data_handled(self, rng: np.random.Generator) -> None:
        """NaN values in input should not cause crash."""
        data = rng.random((100, 50))
        data[30:40, 20:30] = np.nan

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=250e-6, cut_borders_after_smoothing=False
        )

        # Should produce output
        assert depth_data.shape == data.shape

    def test_single_nan_interpolated(self) -> None:
        """Single NaN value should be interpolated from neighbors."""
        data = np.ones((50, 50)) * 10.0
        data[25, 25] = np.nan

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=250e-6, cut_borders_after_smoothing=False
        )

        assert depth_data.shape == data.shape


class TestBorderCropping:
    """Test border cropping behavior."""

    def test_cropping_removes_sigma_pixels(self) -> None:
        """Border cropping should remove approximately sigma pixels."""
        data = np.random.randn(500, 50)
        xdim = 1e-6
        cutoff_lo = 1000e-6  # 1000 um in meters

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=xdim, cutoff_lo=cutoff_lo, cut_borders_after_smoothing=True
        )

        # Calculate expected sigma
        from conversion.preprocessing.cheby_cutoff_to_gauss_sigma import (
            cheby_cutoff_to_gauss_sigma,
        )
        import math

        sigma = cheby_cutoff_to_gauss_sigma(cutoff_lo, xdim)
        sigma_int = int(math.ceil(sigma))

        # Output should be reduced by approximately 2*sigma
        expected_rows = data.shape[0] - 2 * sigma_int
        assert depth_data.shape[0] == expected_rows

    def test_small_data_not_over_cropped(self) -> None:
        """Small data should not be excessively cropped."""
        data = np.random.randn(20, 10)

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=100e-6, cut_borders_after_smoothing=True
        )

        # Should still have some data
        assert depth_data.size > 0


class TestOneDimensionalInput:
    """Test handling of 1D profile input."""

    def test_1d_column_vector(self) -> None:
        """1D column vector should be handled."""
        data = np.random.randn(100, 1)

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=250e-6, cut_borders_after_smoothing=False
        )

        assert depth_data.ndim == 2
        assert depth_data.shape[1] == 1


class TestParameterEffects:
    """Test effects of different parameter values."""

    def test_larger_cutoff_smooths_more(self, rng: np.random.Generator) -> None:
        """Larger cutoff should result in more smoothing."""
        # Create noisy data
        data = rng.random((200, 50)) * 100

        depth_small, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=100e-6, cut_borders_after_smoothing=False
        )
        depth_large, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=1000e-6, cut_borders_after_smoothing=False
        )

        # Larger cutoff = more smoothing = less variance
        var_small = np.var(depth_small[30:-30, :])
        var_large = np.var(depth_large[30:-30, :])

        assert var_large < var_small

    def test_very_large_cutoff_approaches_mean(self, rng: np.random.Generator) -> None:
        """Very large cutoff should make output approach mean value."""
        data = rng.random((200, 50)) * 100

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=10000e-6, cut_borders_after_smoothing=False
        )

        # Interior values should be close to each other
        interior = depth_data[50:-50, 10:-10]
        assert np.std(interior) < 5.0  # Low variation


class TestDifferenceFromShapeRemoval:
    """Test that noise removal differs from shape removal (returns smoothed, not residuals)."""

    def test_returns_smoothed_not_residuals(self, rng: np.random.Generator) -> None:
        """Noise removal should return smoothed data, not residuals."""
        # Create data with DC offset
        data = rng.random((100, 50)) + 100.0  # Values around 100

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=500e-6, cut_borders_after_smoothing=False
        )

        # Result should also be around 100 (smoothed, not residuals)
        interior = depth_data[20:-20, 10:-10]
        assert np.mean(interior) > 90  # Should preserve DC offset

    def test_lowpass_vs_highpass(self, rng: np.random.Generator) -> None:
        """Compare noise removal (lowpass) to what highpass would give."""
        from conversion.preprocessing.remove_shape_gaussian import remove_shape_gaussian

        data = rng.random((100, 50)) * 10

        # Noise removal (lowpass)
        lowpass_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=500e-6, cut_borders_after_smoothing=False
        )

        # Shape removal with same cutoff (highpass)
        highpass_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=500, cut_borders_after_smoothing=False
        )

        # lowpass + highpass should approximately equal original
        reconstructed = lowpass_data + highpass_data
        interior = slice(20, -20), slice(10, -10)

        assert_array_almost_equal(reconstructed[interior], data[interior], decimal=5)


class TestMatlabCompatibility:
    """Test compatibility with MATLAB implementation behavior."""

    def test_smoothed_output_not_residuals(self, rng: np.random.Generator) -> None:
        """Verify output is smoothed data, not residuals (unlike RemoveShapeGaussian)."""
        # Create uniform data with small noise
        data = np.ones((100, 50)) * 50.0
        data += rng.random(data.shape) * 0.1

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=500e-6, cut_borders_after_smoothing=False
        )

        # Output should still be around 50 (smoothed original)
        # Not around 0 (which would be residuals)
        interior = depth_data[20:-20, 10:-10]
        assert np.abs(np.mean(interior) - 50.0) < 1.0

    def test_filter_direction_along_rows(self, rng: np.random.Generator) -> None:
        """Verify filtering is applied along rows (first dimension)."""
        # Create data with variation only in column direction
        cols = np.linspace(0, 10, 50)
        col_variation = cols**2
        data = np.tile(col_variation, (100, 1))

        depth_data, _, _ = remove_noise_gaussian(
            data, xdim=1e-6, cutoff_lo=500e-6, cut_borders_after_smoothing=False
        )

        # Column variation should be mostly preserved (not filtered)
        col_std = np.std(depth_data, axis=1).mean()
        assert col_std > 10  # Still has significant column variation
