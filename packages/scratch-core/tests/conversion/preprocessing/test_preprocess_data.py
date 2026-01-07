"""Tests for form and noise removal pipeline (Step 2 of PreprocessData).

This module tests the integration of shape removal and noise removal,
including the short data handling and slope correction paths.
"""

import numpy as np
import pytest

from conversion.preprocess_data import (
    FormNoiseRemovalResult,
    apply_form_noise_removal,
)
from conversion.cheby_cutoff_to_gauss_sigma import cheby_cutoff_to_gauss_sigma

SEED: int = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(SEED)


class TestFormNoiseRemovalResult:
    """Test the FormNoiseRemovalResult dataclass."""

    def test_result_has_expected_attributes(self) -> None:
        """Verify result dataclass has all expected attributes."""
        depth_data = np.zeros((10, 10))
        mask = np.ones((10, 10), dtype=bool)

        result = FormNoiseRemovalResult(
            depth_data=depth_data,
            mask=mask,
            relative_highest_point_location=None,
        )

        assert hasattr(result, "depth_data")
        assert hasattr(result, "mask")
        assert hasattr(result, "relative_highest_point_location")

    def test_result_with_highest_point(self) -> None:
        """Verify result can store highest point location."""
        result = FormNoiseRemovalResult(
            depth_data=np.zeros((10, 10)),
            mask=np.ones((10, 10), dtype=bool),
            relative_highest_point_location=0.5,
        )

        assert result.relative_highest_point_location == 0.5


class TestApplyFormNoiseRemovalBasic:
    """Test basic functionality of apply_form_noise_removal."""

    def test_produces_valid_output(self, rng: np.random.Generator) -> None:
        """Function should produce valid output without errors."""
        data = rng.random((500, 100))

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000)

        assert result.depth_data is not None
        assert result.mask is not None
        assert result.depth_data.ndim == 2

    def test_output_shape_changes_with_cropping(self, rng: np.random.Generator) -> None:
        """Output shape should be reduced due to border cropping."""
        data = rng.random((500, 100))

        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, cut_borders_after_smoothing=True
        )

        # Output should be smaller than input
        assert result.depth_data.shape[0] < data.shape[0]

    def test_output_shape_preserved_no_cropping(self, rng: np.random.Generator) -> None:
        """Output shape should match input when cropping disabled."""
        data = rng.random((500, 100))

        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, cut_borders_after_smoothing=False
        )

        assert result.depth_data.shape == data.shape

    def test_mask_shape_matches_output(self, rng: np.random.Generator) -> None:
        """Output mask shape should match output data shape."""
        data = rng.random((200, 50))

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000)

        assert result.mask.shape == result.depth_data.shape


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
        sigma = cheby_cutoff_to_gauss_sigma(2000, 1e-6)
        # Short data: 2*sigma > height * 0.2
        # height < 2*sigma / 0.2 = 10*sigma
        short_height = int(5 * sigma)  # Clearly below threshold

        data = np.random.randn(short_height, 50)

        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, cut_borders_after_smoothing=True
        )

        # Output should have same shape as input (cropping disabled)
        assert result.depth_data.shape == data.shape

    def test_long_data_enables_cropping(self) -> None:
        """Long data should allow border cropping."""
        # Create long data where 2*sigma <= 20% of height
        sigma = cheby_cutoff_to_gauss_sigma(2000, 1e-6)
        long_height = int(20 * sigma)  # Clearly above threshold

        data = np.random.randn(long_height, 50)

        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, cut_borders_after_smoothing=True
        )

        # Output should be smaller than input (cropping enabled)
        assert result.depth_data.shape[0] < data.shape[0]


class TestSlopeCorrection:
    """Test slope correction path (UnfoldBullet)."""

    def test_slope_correction_false_no_highest_point(
        self, rng: np.random.Generator
    ) -> None:
        """Without slope correction, highest point should be None."""
        data = rng.random((500, 100))

        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, slope_correction=False
        )

        assert result.relative_highest_point_location is None

    def test_slope_correction_true_returns_highest_point(
        self, rng: np.random.Generator
    ) -> None:
        """With slope correction, highest point should be returned."""
        data = rng.random((500, 100))

        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, slope_correction=True
        )

        assert result.relative_highest_point_location is not None
        assert 0 <= result.relative_highest_point_location <= 1

    def test_slope_correction_uses_unfold_bullet(self) -> None:
        """Slope correction should use UnfoldBullet for curved surfaces."""
        # Create curved bullet surface
        rows = np.linspace(-10, 10, 500)
        curvature = -0.05 * rows**2
        data = np.tile(curvature.reshape(-1, 1), (1, 100))

        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, slope_correction=True
        )

        # Should have highest point near center for symmetric curvature
        assert 0.3 < result.relative_highest_point_location < 0.7


class TestMaskHandling:
    """Test handling of input masks."""

    def test_no_mask_works(self, rng: np.random.Generator) -> None:
        """Function should work without mask."""
        data = rng.random((200, 50))

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000, mask=None)

        assert result.depth_data is not None

    def test_with_mask_works(self, rng: np.random.Generator) -> None:
        """Function should work with mask."""
        data = rng.random((200, 50))
        mask = np.ones(data.shape, dtype=bool)
        mask[50:100, 20:30] = False

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000, mask=mask)

        assert result.depth_data is not None


class TestParameterEffects:
    """Test effects of different parameters."""

    def test_different_cutoff_hi_values(self, rng: np.random.Generator) -> None:
        """Different cutoff_hi should affect shape removal."""
        data = rng.random((500, 100))

        result_small = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=1000)
        result_large = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=5000)

        # Both should produce valid output
        assert result_small.depth_data.shape[0] > 0
        assert result_large.depth_data.shape[0] > 0

        # Larger cutoff means more cropping
        assert result_large.depth_data.shape[0] <= result_small.depth_data.shape[0]

    def test_different_cutoff_lo_values(self, rng: np.random.Generator) -> None:
        """Different cutoff_lo should affect noise removal."""
        data = rng.random((500, 100))

        result_small = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, cutoff_lo=100
        )
        result_large = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, cutoff_lo=1000
        )

        # Both should produce valid output
        assert result_small.depth_data.shape[0] > 0
        assert result_large.depth_data.shape[0] > 0


class TestMatlabCompatibility:
    """Test compatibility with MATLAB PreprocessData.m lines 156-192."""

    def test_replicates_short_data_check(self) -> None:
        """Verify the short data check matches MATLAB logic.

        MATLAB: if 2 * sigma > size(data_rot.depth_data, 1) * 0.2
        """
        xdim = 1e-6
        cutoff_hi = 2000.0

        sigma = cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)

        # Threshold height where 2*sigma = height * 0.2
        threshold_height = int(2 * sigma / 0.2)

        # Data just below threshold
        short_data = np.random.randn(threshold_height - 10, 50)
        result_short = apply_form_noise_removal(
            short_data, xdim=xdim, cutoff_hi=cutoff_hi, cut_borders_after_smoothing=True
        )

        # Data just above threshold
        long_data = np.random.randn(threshold_height + 100, 50)
        result_long = apply_form_noise_removal(
            long_data, xdim=xdim, cutoff_hi=cutoff_hi, cut_borders_after_smoothing=True
        )

        # Short data should not be cropped
        assert result_short.depth_data.shape == short_data.shape

        # Long data should be cropped
        assert result_long.depth_data.shape[0] < long_data.shape[0]

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

        result = apply_form_noise_removal(
            data,
            xdim=1e-6,
            cutoff_hi=2000,
            cutoff_lo=250,
            cut_borders_after_smoothing=False,
        )

        # Output should have reduced shape component
        # Check by looking at variance reduction
        output_var = np.var(result.depth_data)
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
        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, cutoff_lo=250
        )

        # Verify output
        assert result.depth_data.ndim == 2
        assert result.depth_data.shape[1] == n_cols
        assert result.relative_highest_point_location is None  # No slope correction

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
        result = apply_form_noise_removal(
            data, xdim=1e-6, cutoff_hi=2000, slope_correction=True
        )

        # Verify output
        assert result.depth_data.ndim == 2
        assert result.depth_data.shape[1] == n_cols
        assert result.relative_highest_point_location is not None
        # Highest point should be near center for symmetric curvature
        assert 0.3 < result.relative_highest_point_location < 0.7


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_data(self) -> None:
        """Small data should be handled without crash."""
        data = np.random.randn(50, 20)

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=100)

        assert result.depth_data is not None

    def test_single_column(self) -> None:
        """Single column data should be handled."""
        data = np.random.randn(500, 1)

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000)

        assert result.depth_data.shape[1] == 1

    def test_1d_input(self) -> None:
        """1D input should be handled."""
        data = np.random.randn(500)

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000)

        assert result.depth_data.ndim == 2

    def test_data_with_nan(self) -> None:
        """Data with NaN values should not crash."""
        data = np.random.randn(500, 100)
        data[200:250, 40:60] = np.nan

        result = apply_form_noise_removal(data, xdim=1e-6, cutoff_hi=2000)

        assert result is not None
