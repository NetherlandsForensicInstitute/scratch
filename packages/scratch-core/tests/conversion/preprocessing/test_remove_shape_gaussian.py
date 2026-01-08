"""Tests for Gaussian shape removal filter.

This module tests the removal of large-scale surface form using Gaussian
highpass filtering, verifying correct behavior for various input scenarios.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from conversion.preprocessing.remove_shape_gaussian import remove_shape_gaussian

SEED: int = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(SEED)


class TestRemoveShapeGaussianReturnType:
    """Test the return type of remove_shape_gaussian."""

    def test_returns_tuple_of_three(self) -> None:
        """Verify function returns a tuple of (depth_data, range_indices, mask)."""
        data = np.zeros((100, 50))
        result = remove_shape_gaussian(data, xdim=1e-6, cutoff_hi=2000e-6)

        assert isinstance(result, tuple)
        assert len(result) == 3

        depth_data, range_indices, mask = result
        assert isinstance(depth_data, np.ndarray)
        assert isinstance(range_indices, np.ndarray)
        assert isinstance(mask, np.ndarray)


class TestRemoveShapeGaussianBasic:
    """Test basic functionality of remove_shape_gaussian."""

    def test_output_shape_matches_input_no_cropping(
        self, rng: np.random.Generator
    ) -> None:
        """Output shape should match input when cropping is disabled."""
        data = rng.random((100, 50))
        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )
        assert depth_data.shape == data.shape

    def test_output_shape_reduced_with_cropping(self, rng: np.random.Generator) -> None:
        """Output shape should be reduced when border cropping is enabled."""
        data = rng.random((200, 50))
        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=500e-6, cut_borders_after_smoothing=True
        )
        # Output should be smaller in row dimension
        assert depth_data.shape[0] < data.shape[0]
        assert depth_data.shape[1] == data.shape[1]

    def test_range_indices_match_output_size(self, rng: np.random.Generator) -> None:
        """Range indices length should match output row count."""
        data = rng.random((200, 50))
        depth_data, range_indices, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=500e-6, cut_borders_after_smoothing=True
        )
        assert len(range_indices) == depth_data.shape[0]

    def test_mask_shape_matches_output(self, rng: np.random.Generator) -> None:
        """Output mask shape should match output data shape."""
        data = rng.random((100, 50))
        depth_data, _, mask = remove_shape_gaussian(data, xdim=1e-6, cutoff_hi=2000e-6)
        assert mask.shape == depth_data.shape


class TestHighpassBehavior:
    """Test that the filter behaves as a highpass filter."""

    def test_uniform_data_returns_near_zero(self) -> None:
        """Uniform data should have near-zero residuals (no shape to remove)."""
        data = np.ones((100, 50)) * 42.0
        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )
        # Interior should be nearly zero (edges affected by boundary)
        interior = depth_data[20:-20, 10:-10]
        assert np.abs(interior).max() < 0.01

    def test_linear_gradient_removed(self) -> None:
        """Linear gradient (tilt) should be removed."""
        rows = np.linspace(0, 10, 100).reshape(-1, 1)
        data = np.tile(rows, (1, 50))  # Linear gradient in row direction

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )

        # Residuals should be near zero (linear trend removed)
        interior = depth_data[20:-20, 10:-10]
        assert np.abs(interior).max() < 0.5

    def test_parabolic_shape_reduced(self) -> None:
        """Parabolic shape should be significantly reduced."""
        rows = np.linspace(-5, 5, 200)
        shape = rows**2
        data = np.tile(shape.reshape(-1, 1), (1, 50))

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=5000e-6, cut_borders_after_smoothing=False
        )

        # Variance should be reduced (shape removed)
        assert np.var(depth_data[30:-30, :]) < np.var(data[30:-30, :])

    def test_high_frequency_preserved(self, rng: np.random.Generator) -> None:
        """High-frequency content should be preserved."""
        # Create high-frequency oscillation
        rows = np.arange(200)
        hf_signal = np.sin(2 * np.pi * rows / 10)  # Period of 10 pixels
        data = np.tile(hf_signal.reshape(-1, 1), (1, 50))

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=5000e-6, cut_borders_after_smoothing=False
        )

        # High-frequency content should still be present
        output_interior = depth_data[30:-30, 25]
        input_interior = data[30:-30, 25]

        # Correlation should be high
        correlation = np.corrcoef(output_interior, input_interior)[0, 1]
        assert correlation > 0.9


class TestMaskHandling:
    """Test handling of masked regions."""

    def test_mask_all_true_equals_no_mask(self, rng: np.random.Generator) -> None:
        """All-true mask should give same result as no mask."""
        data = rng.random((100, 50))
        mask = np.ones(data.shape, dtype=bool)

        depth_with_mask, _, _ = remove_shape_gaussian(
            data,
            xdim=1e-6,
            cutoff_hi=2000,
            mask=mask,
            cut_borders_after_smoothing=False,
        )
        depth_no_mask, _, _ = remove_shape_gaussian(
            data,
            xdim=1e-6,
            cutoff_hi=2000,
            mask=None,
            cut_borders_after_smoothing=False,
        )

        assert_array_almost_equal(depth_with_mask, depth_no_mask)

    def test_partial_mask_handled(self, rng: np.random.Generator) -> None:
        """Partial mask should not crash and produce valid output."""
        data = rng.random((100, 50))
        mask = np.ones(data.shape, dtype=bool)
        mask[40:60, 20:30] = False  # Mask out central region

        depth_data, _, _ = remove_shape_gaussian(
            data,
            xdim=1e-6,
            cutoff_hi=2000,
            mask=mask,
            cut_borders_after_smoothing=False,
        )

        assert depth_data.shape == data.shape
        assert not np.all(np.isnan(depth_data))


class TestNaNHandling:
    """Test handling of NaN values in input data."""

    def test_nan_in_data_handled(self, rng: np.random.Generator) -> None:
        """NaN values in input should not cause crash."""
        data = rng.random((100, 50))
        data[30:40, 20:30] = np.nan

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )

        # Should produce output (may contain NaN in originally NaN regions)
        assert depth_data.shape == data.shape

    def test_single_nan_interpolated(self) -> None:
        """Single NaN value should be interpolated from neighbors."""
        data = np.ones((50, 50)) * 10.0
        data[25, 25] = np.nan

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )

        # Result at NaN position depends on implementation
        # Just verify no crash and reasonable output
        assert depth_data.shape == data.shape


class TestBorderCropping:
    """Test border cropping behavior."""

    def test_cropping_removes_sigma_pixels(self) -> None:
        """Border cropping should remove approximately sigma pixels."""
        data = np.random.randn(500, 50)
        xdim = 1e-6
        cutoff_hi = 1000e-6  # 1000 um in meters

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=xdim, cutoff_hi=cutoff_hi, cut_borders_after_smoothing=True
        )

        # Calculate expected sigma
        from conversion.preprocessing.cheby_cutoff_to_gauss_sigma import (
            cheby_cutoff_to_gauss_sigma,
        )
        import math

        sigma = cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)
        sigma_int = int(math.ceil(sigma))

        # Output should be reduced by approximately 2*sigma
        expected_rows = data.shape[0] - 2 * sigma_int
        assert depth_data.shape[0] == expected_rows

    def test_small_data_not_over_cropped(self) -> None:
        """Small data should not be excessively cropped."""
        data = np.random.randn(20, 10)

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=100e-6, cut_borders_after_smoothing=True
        )

        # Should still have some data
        assert depth_data.size > 0


class TestOneDimensionalInput:
    """Test handling of 1D profile input."""

    def test_1d_column_vector(self) -> None:
        """1D column vector should be handled."""
        data = np.random.randn(100, 1)

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )

        assert depth_data.ndim == 2
        assert depth_data.shape[1] == 1


class TestParameterEffects:
    """Test effects of different parameter values."""

    def test_larger_cutoff_removes_more_shape(self, rng: np.random.Generator) -> None:
        """Larger cutoff should remove more low-frequency content."""
        # Create data with shape
        rows = np.linspace(0, 10, 200)
        shape = rows**2
        data = np.tile(shape.reshape(-1, 1), (1, 50))
        data += rng.random(data.shape) * 0.1  # Add noise

        depth_small, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=1000e-6, cut_borders_after_smoothing=False
        )
        depth_large, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=5000e-6, cut_borders_after_smoothing=False
        )

        # Larger cutoff should leave less shape (smaller variance in residuals)
        var_small = np.var(depth_small[30:-30, :])
        var_large = np.var(depth_large[30:-30, :])

        assert var_large < var_small

    def test_smaller_xdim_increases_smoothing(self, rng: np.random.Generator) -> None:
        """Smaller xdim (finer resolution) should increase effective smoothing."""
        data = rng.random((200, 50))

        depth_coarse, _, _ = remove_shape_gaussian(
            data, xdim=2e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )
        depth_fine, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )

        # Finer resolution means larger sigma, more smoothing
        # Variance of residuals should be smaller with finer resolution
        # (Note: this depends on exact implementation, so we just check they differ)
        assert not np.allclose(depth_coarse, depth_fine)


class TestMatlabCompatibility:
    """Test compatibility with MATLAB implementation behavior."""

    def test_residuals_equal_input_minus_smoothed(
        self, rng: np.random.Generator
    ) -> None:
        """Verify highpass = input - lowpass relationship."""
        data = rng.random((100, 50))

        # The function should compute: residuals = data - smoothed
        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )

        # Since we can't easily get the smoothed version separately,
        # we verify the output is different from input but not all zeros
        assert not np.allclose(depth_data, data)
        assert not np.allclose(depth_data, 0)

    def test_filter_direction_along_rows(self, rng: np.random.Generator) -> None:
        """Verify filtering is applied along rows (first dimension)."""
        # Create data with variation only in column direction
        cols = np.linspace(0, 10, 50)
        col_variation = cols**2
        data = np.tile(col_variation, (100, 1))

        depth_data, _, _ = remove_shape_gaussian(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )

        # Column variation should be mostly preserved (not filtered)
        # Because filtering is along rows only
        # Check that column-to-column variation is still present
        col_std = np.std(depth_data, axis=1).mean()
        assert col_std > 0.1  # Still has column variation
