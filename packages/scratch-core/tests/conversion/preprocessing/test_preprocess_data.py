"""Tests for form and noise removal pipeline (Step 2 of PreprocessData).

This module tests the integration of shape removal and noise removal,
including the short data handling and slope correction paths.
"""

import numpy as np
import pytest

from conversion.preprocessing.preprocess_data import apply_form_noise_removal
from conversion.preprocessing.cheby_cutoff_to_gauss_sigma import (
    cheby_cutoff_to_gauss_sigma,
)

SEED: int = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(SEED)


class TestApplyFormNoiseRemovalReturnType:
    """Test the return type of apply_form_noise_removal."""

    def test_returns_tuple_of_three(self) -> None:
        """Verify function returns a tuple of (depth_data, mask, highest_point)."""
        data = np.random.randn(500, 100)
        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000e-6)

        assert isinstance(result, tuple)
        assert len(result) == 3

        depth_data, mask, highest_point = result
        assert isinstance(depth_data, np.ndarray)
        assert isinstance(mask, np.ndarray)
        # highest_point is None when slope_correction=False
        assert highest_point is None

    def test_returns_highest_point_with_slope_correction(self) -> None:
        """Verify highest_point is returned with slope_correction=True."""
        data = np.random.randn(500, 100)
        depth_data, mask, highest_point = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, slope_correction=True
        )

        assert isinstance(highest_point, float)
        assert 0 <= highest_point <= 1


class TestApplyFormNoiseRemovalBasic:
    """Test basic functionality of apply_form_noise_removal."""

    def test_produces_valid_output(self, rng: np.random.Generator) -> None:
        """Function should produce valid output without errors."""
        data = rng.random((500, 100))

        depth_data, mask, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6
        )

        assert depth_data is not None
        assert mask is not None
        assert depth_data.ndim == 2

    def test_output_shape_changes_with_cropping(self, rng: np.random.Generator) -> None:
        """Output shape should be reduced due to border cropping."""
        data = rng.random((500, 100))

        depth_data, _, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=True
        )

        # Output should be smaller than input
        assert depth_data.shape[0] < data.shape[0]

    def test_output_shape_preserved_no_cropping(self, rng: np.random.Generator) -> None:
        """Output shape should match input when cropping disabled."""
        data = rng.random((500, 100))

        depth_data, _, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=False
        )

        assert depth_data.shape == data.shape

    def test_mask_shape_matches_output(self, rng: np.random.Generator) -> None:
        """Output mask shape should match output data shape."""
        data = rng.random((200, 50))

        depth_data, mask, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6
        )

        assert mask.shape == depth_data.shape


class TestShortDataHandling:
    """Test automatic handling of short data."""

    def test_short_data_disables_cropping(self) -> None:
        """Short data should automatically disable border cropping.

        MATLAB: if 2 * sigma > size(data_rot.depth_data, 1) * 0.2
        """
        # Create short data where 2*sigma > 20% of height
        # With xdim=1e-6 and cutoff_hi=2000, sigma ≈ 374
        # For 2*sigma > height*0.2, need height < 2*374/0.2 = 3740
        # Use smaller data to ensure this triggers

        # Calculate sigma to determine short data threshold
        sigma = cheby_cutoff_to_gauss_sigma(2000e-6, 1e-6)
        # Short data: 2*sigma > height * 0.2
        # height < 2*sigma / 0.2 = 10*sigma
        short_height = int(5 * sigma)  # Clearly below threshold

        data = np.random.randn(short_height, 50)

        depth_data, _, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=True
        )

        # Output should have same shape as input (cropping disabled)
        assert depth_data.shape == data.shape

    def test_long_data_enables_cropping(self) -> None:
        """Long data should allow border cropping."""
        # Create long data where 2*sigma <= 20% of height
        sigma = cheby_cutoff_to_gauss_sigma(2000e-6, 1e-6)
        long_height = int(20 * sigma)  # Clearly above threshold

        data = np.random.randn(long_height, 50)

        depth_data, _, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, cut_borders_after_smoothing=True
        )

        # Output should be smaller than input (cropping enabled)
        assert depth_data.shape[0] < data.shape[0]


class TestSlopeCorrection:
    """Test slope correction path (UnfoldBullet)."""

    def test_slope_correction_false_no_highest_point(
        self, rng: np.random.Generator
    ) -> None:
        """Without slope correction, highest point should be None."""
        data = rng.random((500, 100))

        _, _, highest_point = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, slope_correction=False
        )

        assert highest_point is None

    def test_slope_correction_true_returns_highest_point(
        self, rng: np.random.Generator
    ) -> None:
        """With slope correction, highest point should be returned."""
        data = rng.random((500, 100))

        _, _, highest_point = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, slope_correction=True
        )

        assert highest_point is not None
        assert 0 <= highest_point <= 1

    def test_slope_correction_uses_unfold_bullet(self) -> None:
        """Slope correction should use UnfoldBullet for curved surfaces."""
        # Create curved bullet surface
        rows = np.linspace(-10, 10, 500)
        curvature = -0.05 * rows**2
        data = np.tile(curvature.reshape(-1, 1), (1, 100))

        _, _, highest_point = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, slope_correction=True
        )

        # Should have highest point near center for symmetric curvature
        assert 0.3 < highest_point < 0.7


class TestMaskHandling:
    """Test handling of input masks."""

    def test_no_mask_works(self, rng: np.random.Generator) -> None:
        """Function should work without mask."""
        data = rng.random((200, 50))

        depth_data, _, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, mask=None
        )

        assert depth_data is not None

    def test_with_mask_works(self, rng: np.random.Generator) -> None:
        """Function should work with mask."""
        data = rng.random((200, 50))
        mask = np.ones(data.shape, dtype=bool)
        mask[50:100, 20:30] = False

        depth_data, _, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, mask=mask
        )

        assert depth_data is not None


class TestParameterEffects:
    """Test effects of different parameters."""

    def test_different_cutoff_hi_values(self, rng: np.random.Generator) -> None:
        """Different cutoff_hi should affect shape removal."""
        data = rng.random((500, 100))

        depth_small, _, _ = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=1000e-6)
        depth_large, _, _ = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=5000e-6)

        # Both should produce valid output
        assert depth_small.shape[0] > 0
        assert depth_large.shape[0] > 0

        # Larger cutoff means more cropping
        assert depth_large.shape[0] <= depth_small.shape[0]

    def test_different_cutoff_lo_values(self, rng: np.random.Generator) -> None:
        """Different cutoff_lo should affect noise removal."""
        data = rng.random((500, 100))

        depth_small, _, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, cutoff_lo=100e-6
        )
        depth_large, _, _ = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, cutoff_lo=1000e-6
        )

        # Both should produce valid output
        assert depth_small.shape[0] > 0
        assert depth_large.shape[0] > 0


class TestMatlabCompatibility:
    """Test compatibility with MATLAB PreprocessData.m lines 156-192."""

    def test_replicates_short_data_check(self) -> None:
        """Verify the short data check matches MATLAB logic.

        MATLAB: if 2 * sigma > size(data_rot.depth_data, 1) * 0.2
        """
        xdim = 1e-6
        cutoff_hi = 2000e-6  # 2000 um in meters

        sigma = cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)

        # Threshold height where 2*sigma = height * 0.2
        threshold_height = int(2 * sigma / 0.2)

        # Data just below threshold
        short_data = np.random.randn(threshold_height - 10, 50)
        depth_short, _, _ = apply_form_noise_removal(
            short_data, xdim=xdim, cutoff_hi=cutoff_hi, cut_borders_after_smoothing=True
        )

        # Data just above threshold
        long_data = np.random.randn(threshold_height + 100, 50)
        depth_long, _, _ = apply_form_noise_removal(
            long_data, xdim=xdim, cutoff_hi=cutoff_hi, cut_borders_after_smoothing=True
        )

        # Short data should not be cropped
        assert depth_short.shape == short_data.shape

        # Long data should be cropped
        assert depth_long.shape[0] < long_data.shape[0]

    def test_shape_then_noise_order(self, rng: np.random.Generator) -> None:
        """Verify shape removal happens before noise removal.

        The order matters: shape removal produces highpass residuals,
        then noise removal smooths those residuals.
        """
        # Create data with distinct shape and noise components
        rows = np.linspace(0, 100, 500)
        shape = 0.01 * rows**2  # Large-scale shape
        noise = rng.random(500) * 0.001  # Fine noise

        data = np.tile((shape + noise).reshape(-1, 1), (1, 50))

        depth_data, _, _ = apply_form_noise_removal(
            data,
            xdim=1e-6,
            cutoff_hi=2000e-6,
            cutoff_lo=250e-6,
            cut_borders_after_smoothing=False,
        )

        # Output should have reduced shape component
        # Check by looking at variance reduction
        output_var = np.var(depth_data)
        input_var = np.var(data)

        # Variance should be reduced (shape removed, noise smoothed)
        assert output_var < input_var


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_synthetic_toolmark(self, rng: np.random.Generator) -> None:
        """Test full pipeline with synthetic toolmark surface."""
        n_rows, n_cols = 500, 100

        # Create synthetic toolmark
        rows = np.linspace(0, 100, n_rows)

        # Shape component (curvature)
        shape = 0.001 * (rows - 50) ** 2

        # Striation component (periodic)
        striations = 0.01 * np.sin(2 * np.pi * rows / 10)

        # Noise component
        noise = rng.random(n_rows) * 0.001

        # Combine
        surface = shape + striations + noise
        data = np.tile(surface.reshape(-1, 1), (1, n_cols))

        # Apply form and noise removal
        depth_data, _, highest_point = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, cutoff_lo=250e-6
        )

        # Verify output
        assert depth_data.ndim == 2
        assert depth_data.shape[1] == n_cols
        assert highest_point is None  # No slope correction

    def test_full_pipeline_synthetic_bullet(self, rng: np.random.Generator) -> None:
        """Test full pipeline with synthetic bullet surface."""
        n_rows, n_cols = 500, 100

        # Create synthetic bullet
        rows = np.linspace(-10, 10, n_rows)

        # Curvature (convex bullet shape)
        curvature = -0.05 * rows**2

        # Striations
        striations = 0.001 * np.sin(2 * np.pi * rows * 2)

        # Noise
        noise = rng.random(n_rows) * 0.0001

        # Combine
        surface = curvature + striations + noise
        data = np.tile(surface.reshape(-1, 1), (1, n_cols))

        # Apply with slope correction
        depth_data, _, highest_point = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000e-6, slope_correction=True
        )

        # Verify output
        assert depth_data.ndim == 2
        assert depth_data.shape[1] == n_cols
        assert highest_point is not None
        # Highest point should be near center for symmetric curvature
        assert 0.3 < highest_point < 0.7


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_data(self) -> None:
        """Small data should be handled without crash."""
        data = np.random.randn(50, 20)

        depth_data, _, _ = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=100e-6)

        assert depth_data is not None

    def test_single_column(self) -> None:
        """Single column data should be handled."""
        data = np.random.randn(500, 1)

        depth_data, _, _ = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000e-6)

        assert depth_data.shape[1] == 1

    def test_data_with_nan(self) -> None:
        """Data with NaN values should not crash."""
        data = np.random.randn(500, 100)
        data[200:250, 40:60] = np.nan

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000e-6)

        assert result is not None
