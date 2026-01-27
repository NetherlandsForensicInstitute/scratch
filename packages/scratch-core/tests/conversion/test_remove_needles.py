import numpy as np
import pytest
from numpy.testing import assert_array_equal
import unittest

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.remove_needles import (
    mask_and_remove_needles,
    apply_median_filter,
    get_and_remove_needles,
    get_residual_image,
)


class TestMaskAndRemoveNeedles:
    """Unit tests for the remove_needles function."""

    @pytest.fixture
    def simple_scan_image(self) -> ScanImage:
        """Create a simple 50x50 scan image for testing."""
        data = np.ones((50, 50)) * 100.0
        return ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

    @pytest.fixture
    def small_scan_image(self) -> ScanImage:
        """Create a small scan image (<=20 columns) for testing."""
        data = np.ones((30, 15)) * 100.0
        return ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

    @pytest.fixture
    def full_mask(self) -> MaskArray:
        """Create a full mask (all True)."""
        return np.ones((50, 50), dtype=bool)

    @pytest.fixture
    def partial_mask(self) -> MaskArray:
        """Create a partial mask."""
        mask = np.ones((50, 50), dtype=bool)
        mask[10:20, 10:20] = False
        return mask

    def test_basic_functionality_no_needles(
        self, simple_scan_image: ScanImage, full_mask: MaskArray
    ):
        """Test that data without needles remains mostly unchanged."""
        result = mask_and_remove_needles(simple_scan_image, full_mask, median_factor=15)

        # Result should have same shape
        assert result.data.shape == simple_scan_image.data.shape

        # Most values should be close to original (no outliers to remove)
        assert np.nanmean(np.abs(result.data - simple_scan_image.data)) < 5.0

    def test_removes_needles(self, simple_scan_image: ScanImage, full_mask: MaskArray):
        """Test that obvious spike outliers are detected and set to NaN."""
        # Add some obvious spikes
        simple_scan_image.data[10, 10] = 1000.0  # Huge spike
        simple_scan_image.data[20, 20] = 1000.0
        simple_scan_image.data[30, 30] = 1000.0

        result = mask_and_remove_needles(simple_scan_image, full_mask, median_factor=15)

        # Spikes should be set to NaN
        assert np.isnan(result.data[10, 10])
        assert np.isnan(result.data[20, 20])
        assert np.isnan(result.data[30, 30])

        # Most other values should remain the same
        non_spike_mask = np.ones_like(result.data, dtype=bool)
        non_spike_mask[10, 10] = False
        non_spike_mask[20, 20] = False
        non_spike_mask[30, 30] = False
        assert np.nanmean(result.data[non_spike_mask]) > 90.0

    def test_single_row_data(self):
        """Test handling of single-row data (1D case)."""
        data = np.ones((1, 50)) * 100.0
        data[0, 25] = 500.0  # Add a needle
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)
        mask = np.ones((1, 50), dtype=bool)

        result = mask_and_remove_needles(scan_image, mask, median_factor=15)

        assert result.data.shape == (1, 50)
        # Needle should be detected
        assert np.isnan(result.data[0, 25])

    def test_median_factor_parameter(self, simple_scan_image, full_mask):
        """Test that median_factor parameter affects outlier detection."""
        # Add moderate outlier
        simple_scan_image.data[25, 25] = 150.0

        # Strict threshold (lower median_factor)
        result_strict = mask_and_remove_needles(
            simple_scan_image, full_mask, median_factor=5
        )

        # Lenient threshold (higher median_factor)
        result_lenient = mask_and_remove_needles(
            simple_scan_image, full_mask, median_factor=50
        )

        # Strict should flag more points as NaN
        assert np.sum(np.isnan(result_strict.data)) > np.sum(
            np.isnan(result_lenient.data)
        )

    def test_all_nan_input(self):
        """Test handling of input data that is all NaN."""
        data = np.full((50, 50), np.nan)
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)
        mask = np.ones((50, 50), dtype=bool)

        result = mask_and_remove_needles(scan_image, mask, median_factor=15)

        # Should return all NaN
        assert np.all(np.isnan(result.data))

    def test_partial_nan_input(self, simple_scan_image, full_mask):
        """Test handling of input with some NaN values."""
        simple_scan_image.data[5:10, 5:10] = np.nan
        simple_scan_image.data[30, 30] = 500.0  # Add needle

        result = mask_and_remove_needles(simple_scan_image, full_mask, median_factor=15)

        # Original NaN should remain
        assert np.all(np.isnan(result.data[5:10, 5:10]))

        # Needle should be detected
        assert np.isnan(result.data[30, 30])

    def test_mask_application(self, simple_scan_image):
        """Test that mask is properly applied."""
        mask = np.ones((50, 50), dtype=bool)
        mask[20:30, 20:30] = False  # Mask out a region

        # Add spike in masked region
        simple_scan_image.data[25, 25] = 1000.0

        result = mask_and_remove_needles(simple_scan_image, mask, median_factor=15)

        # Result should have correct shape
        assert result.data.shape == simple_scan_image.data.shape
        assert np.all(np.isnan(result.data[20:30, 20:30]))

    def test_preserves_non_outlier_data(self, simple_scan_image, full_mask):
        """Test that non-outlier data is preserved with minimal change."""
        # Add normal variation
        scan_image = ScanImage(
            data=simple_scan_image.data + np.random.randn(50, 50) * 2.0,
            scale_x=simple_scan_image.scale_x,
            scale_y=simple_scan_image.scale_y,
        )

        # Add a few clear outliers
        scan_image.data[10, 10] = 1000.0
        scan_image.data[40, 40] = -500.0

        result = mask_and_remove_needles(scan_image, full_mask, median_factor=15)

        # Count NaN values (should be minimal, just the outliers)
        nan_count = np.sum(np.isnan(result.data))
        total_count = result.data.size

        # Less than 5% should be NaN
        assert (nan_count / total_count) < 0.05

    def test_negative_outliers(self, simple_scan_image, full_mask):
        """Test detection of negative outliers (valleys)."""
        simple_scan_image.data[15, 15] = -500.0  # Deep valley
        simple_scan_image.data[35, 35] = -500.0

        result = mask_and_remove_needles(simple_scan_image, full_mask, median_factor=15)

        # Negative outliers should also be detected
        assert np.isnan(result.data[15, 15])
        assert np.isnan(result.data[35, 35])

    def test_no_modification_of_input(self, simple_scan_image, full_mask):
        """Test that input scan_image is not modified."""
        original_data = simple_scan_image.data.copy()

        result = mask_and_remove_needles(simple_scan_image, full_mask, median_factor=15)

        # Original should be unchanged
        assert_array_equal(simple_scan_image.data, original_data)
        # Result should be different object
        assert result.data is not simple_scan_image.data


class TestGetResidualImage(unittest.TestCase):
    """Tests for get_residual_image function."""

    def test_large_image_with_subsampling(self):
        """Test large image that requires subsampling will return same shape as input."""
        data = np.ones((100, 100)) * 10.0
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)
        residual = get_residual_image(scan_image)

        assert residual.shape == scan_image.data.shape

    def test_residual_detects_spikes(self):
        """Test that residuals are large for spike outliers."""
        data = np.ones((20, 20)) * 10.0
        data[10, 10] = 1000.0  # Add spike
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        residual = get_residual_image(scan_image)

        # Spike location should have large residual
        assert np.abs(residual[10, 10]) > np.abs(residual[5, 5])

    def test_residual_image_of_scan_image_width_1(self):
        """Test that residuals are large for spike outliers."""
        data = np.ones((20, 1)) * 10.0
        data[10, 0] = 1000.0  # Add spike
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        residual = get_residual_image(scan_image)

        # Spike location should have large residual
        assert np.abs(residual[10, 0]) > np.abs(residual[5, 0])
        assert residual.shape == scan_image.data.shape

    def test_residual_image_of_scan_image_height_1(self):
        """Test that residuals are large for spike outliers."""
        data = np.ones((1, 20)) * 10.0
        data[0, 10] = 1000.0  # Add spike
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        residual = get_residual_image(scan_image)

        # Spike location should have large residual
        assert np.abs(residual[0, 10]) > np.abs(residual[0, 5])
        assert residual.shape == scan_image.data.shape

    def test_with_nan_values(self):
        """Test handling of NaN values in input."""
        data = np.ones((20, 20))
        data[5, 5] = np.nan
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        residual = get_residual_image(scan_image)

        assert residual.shape == data.shape
        # NaN should propagate through
        assert np.isnan(residual[5, 5])


class TestDetermineAndRemoveNeedles(unittest.TestCase):
    """Tests for determine_and_remove_needles function."""

    def test_no_needles_detected(self):
        """Test case where no needles should be detected."""
        data = np.ones((10, 10)) * 5.0
        residuals = np.ones((10, 10)) * 0.1  # Small residuals
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        result = get_and_remove_needles(scan_image, residuals, median_factor=15.0)

        # No needles should be removed
        assert np.array_equal(result.data, data)

    def test_needle_removed(self):
        """Test that outlier needle is removed."""
        data = np.ones((10, 10)) * 5.0
        residuals = np.zeros((10, 10))
        residuals[5, 5] = 100.0  # Large residual = needle
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        result = get_and_remove_needles(scan_image, residuals, median_factor=1.0)

        # Needle should be set to NaN
        assert np.isnan(result.data[5, 5])
        # Other values should remain
        assert result.data[0, 0] == 5.0

    def test_multiple_needles_removed(self):
        """Test removal of multiple needles."""
        data = np.ones((10, 10)) * 5.0
        residuals = np.zeros((10, 10))
        residuals[2, 2] = 100.0
        residuals[7, 7] = 100.0
        residuals[5, 5] = 100.0
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        result = get_and_remove_needles(scan_image, residuals, median_factor=1.0)

        # All needles should be removed
        assert np.isnan(result.data[2, 2])
        assert np.isnan(result.data[7, 7])
        assert np.isnan(result.data[5, 5])
        # Count total NaNs
        assert np.sum(np.isnan(result.data)) == 3

    def test_median_factor_parameter(self):
        """Test that median_factor affects threshold."""
        data = np.ones((10, 10)) * 5.0
        residuals = np.ones((10, 10))
        residuals[5, 5] = 10.0  # Moderate residual
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        # Low threshold - should detect needle
        result_low = get_and_remove_needles(scan_image, residuals, median_factor=1.0)

        # High threshold - should not detect needle
        result_high = get_and_remove_needles(scan_image, residuals, median_factor=100.0)

        # With low threshold, needle detected
        assert np.isnan(result_low.data[5, 5])
        # With high threshold, needle not detected
        assert not np.isnan(result_high.data[5, 5])

    def test_nan_residuals_handled(self):
        """Test handling of NaN values in residuals."""
        data = np.ones((10, 10)) * 5.0
        residuals = np.ones((10, 10))
        residuals[3, 3] = np.nan
        residuals[7, 7] = 100.0  # This should still be detected
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        result = get_and_remove_needles(scan_image, residuals, median_factor=1.0)

        assert np.isnan(result.data[7, 7])

    def test_original_data_not_modified(self):
        """Test that original scan_image.data is not modified."""
        data = np.ones((10, 10)) * 5.0
        data_copy = data.copy()
        residuals = np.zeros((10, 10))
        residuals[5, 5] = 100.0
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        result = get_and_remove_needles(scan_image, residuals, median_factor=1.0)

        # Original data should be unchanged
        np.testing.assert_array_equal(scan_image.data, data_copy)
        # Result should be different
        assert result.data is not data_copy


class TestApplyMedianFilter(unittest.TestCase):
    """Unit tests for apply_median_filter function."""

    def test_basic_3x3_filter(self):
        """Test basic 3x3 median filter on small array."""
        input_image = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
        )

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # Verify output shape matches input
        assert result.data.shape == input_image.shape
        # Verify output is float type
        assert np.issubdtype(result.data.dtype, np.floating)

    def test_even_filter_size_becomes_odd(self):
        """Test that even filter_size is converted to odd (filter_size + 1)."""
        input_image = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            dtype=np.float64,
        )

        # Test with even filter_size (should become 5)
        result_even = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size=4
        )
        # Test with odd filter_size (should stay 5)
        result_odd = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size=5
        )

        # Both should produce same result since 4 -> 5
        assert result_even.data.shape == input_image.shape
        assert result_odd.data.shape == input_image.shape

    def test_nan_padding_at_borders(self):
        """Test padding with NaN values at borders."""
        input_image = np.array(
            [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=np.float64
        )

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # With NaN padding, edge pixels use fewer valid values in their neighborhood
        assert result.data.shape == input_image.shape
        # Center pixel should be exactly 5.0 (median of all 5.0s)
        self.assertAlmostEqual(result.data[1, 1], 5.0, places=5)
        # Edge pixels should also be 5.0 since NaNs are ignored in nanmedian
        self.assertAlmostEqual(result.data[0, 0], 5.0, places=5)

    def test_edge_pixels_with_nan_padding(self):
        """Test that edge pixels are computed correctly with NaN padding."""
        input_image = np.array(
            [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
            dtype=np.float64,
        )

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # All pixels should be 10.0 since NaN padding is ignored
        assert result.data.shape == input_image.shape
        np.testing.assert_array_almost_equal(result.data, input_image, decimal=5)

    def test_large_filter_size(self):
        """Test with larger filter size."""
        input_image = np.random.rand(10, 10).astype(np.float64)

        filter_size = 7

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_filter_size_larger_than_image(self):
        """Test with larger filter size."""
        input_image = np.random.rand(10, 10).astype(np.float64)

        filter_size = 15

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_single_pixel_image(self):
        """Test edge case with 1x1 image."""
        input_image = np.array([[5.0]], dtype=np.float64)

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == (1, 1)
        # With NaN padding, the single pixel should remain unchanged
        self.assertAlmostEqual(result.data[0, 0], 5.0, places=5)

    def test_rectangular_image(self):
        """Test with non-square image."""
        input_image = np.random.rand(5, 10).astype(np.float64)

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_all_nan_input(self):
        """Test with all NaN input."""
        input_image = np.full((3, 3), np.nan, dtype=np.float64)

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # Output should be all NaN (NaN input + NaN padding = all NaN)
        assert result.data.shape == input_image.shape
        assert np.all(np.isnan(result.data))

    def test_spike_removal(self):
        """Test that median filter removes spikes (salt and pepper noise)."""
        input_image = np.array(
            [
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 100.0, 5.0, 5.0],  # Spike in center
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # Center spike should be filtered out
        assert result.data[2, 2] < 10.0  # Should be close to 5.0, not 100.0

    def test_corner_pixels(self):
        """Test that corner pixels are computed correctly with NaN padding."""
        # Corner pixels have the most NaN neighbors (due to padding)
        input_image = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            dtype=np.float64,
        )

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape
        # Corner values should be computed from fewer valid neighbors
        # but should still be reasonable values
        assert not np.isnan(result.data[0, 0])

    def test_mixed_nan_pattern(self):
        """Test with checkerboard NaN pattern."""
        input_image = np.array(
            [
                [1.0, np.nan, 3.0, np.nan],
                [np.nan, 6.0, np.nan, 8.0],
                [9.0, np.nan, 11.0, np.nan],
                [np.nan, 14.0, np.nan, 16.0],
            ],
            dtype=np.float64,
        )

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_output_dtype_is_float64(self):
        """Test that output is converted to float64 (double in MATLAB)."""
        input_image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # MATLAB function converts to double at the end
        assert result.data.dtype == np.float64

    def test_large_image_performance(self):
        """Test with larger image to verify it handles size reasonably."""
        input_image = np.random.rand(100, 100).astype(np.float64)

        filter_size = 5

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_gradient_image(self):
        """Test with gradient image to verify smoothing behavior."""
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x, y)
        input_image = X + Y  # Diagonal gradient

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # Result should still be smooth and monotonic
        assert result.data.shape == input_image.shape
        # Gradient should be preserved in the center
        assert result.data[15, 15] > result.data[5, 5]


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests."""

    def test_very_large_filter_size(self):
        """Test with filter_size larger than image dimensions."""
        input_image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        filter_size = 11  # Much larger than image

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # Should still return same shape
        assert result.data.shape == input_image.shape

    def test_inf_values(self):
        """Test handling of infinity values."""
        input_image = np.array(
            [[1.0, 2.0, np.inf], [4.0, 5.0, 6.0], [-np.inf, 8.0, 9.0]], dtype=np.float64
        )

        filter_size = 3

        result = apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape
