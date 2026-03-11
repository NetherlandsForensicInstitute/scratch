import re
import unittest

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.constants import micro

from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from exceptions import ImageShapeMismatchError
from mutations.filter import Mask


class TestMask2dArray:
    @pytest.fixture
    def scan_image(
        self,
    ):
        return ScanImage(
            data=np.array([[1, 2], [3, 4]], dtype=float), scale_x=1.0, scale_y=1.0
        )

    def test_mask_sets_background_pixels_to_nan(self, scan_image: ScanImage) -> None:
        # Arrange
        mask = np.array([[1, 0], [0, 1]], dtype=bool)
        masking_mutator = Mask(mask=mask)
        # Act
        result = masking_mutator.apply_on_image(scan_image=scan_image)
        # Assert
        assert np.array_equal(
            result.data, np.array([[1, np.nan], [np.nan, 4]]), equal_nan=True
        )

    def test_raises_on_shape_mismatch(self, scan_image: ScanImage) -> None:
        # Arrange
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
        masking_mutator = Mask(mask=mask)
        # Act / Assert
        with pytest.raises(
            ImageShapeMismatchError,
            match=re.escape(
                f"Mask shape: {mask.shape} does not match image shape: {scan_image.data.shape}"
            ),
        ):
            masking_mutator.apply_on_image(scan_image=scan_image)

    def test_full_mask_preserves_all_values(self, scan_image: ScanImage) -> None:
        # Arrange
        mask = np.ones((2, 2), dtype=bool)
        masking_mutator = Mask(mask=mask)
        # Act
        result = masking_mutator.apply_on_image(scan_image=scan_image)
        # Assert
        assert np.array_equal(result.data, scan_image.data, equal_nan=True)

    def test_full_mask_skips_calculation(
        self, scan_image: ScanImage, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        mask = np.ones((2, 2), dtype=bool)
        masking_mutator = Mask(mask=mask)
        # Act
        result = masking_mutator(scan_image=scan_image).unwrap()
        # Assert
        assert np.array_equal(result.data, scan_image.data, equal_nan=True)
        assert (
            "skipping masking, Mask area is not containing any masking fields."
            in caplog.messages
        )

    def test_empty_mask_sets_all_to_nan(
        self, scan_image: ScanImage, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        mask = np.zeros((2, 2), dtype=bool)
        masking_mutator = Mask(mask=mask)
        result = masking_mutator(scan_image=scan_image).unwrap()

        assert np.all(np.isnan(result.data))
        assert "Applying mask to scan_image" in caplog.messages


class TestMaskAndRemoveNeedles:
    """Unit tests for the remove_needles function."""

    @pytest.fixture
    def simple_scan_image(self) -> ScanImage:
        """Create a simple 50x50 scan image for testing."""
        data = np.ones((50, 50)) * 100.0
        return ScanImage(data=data.copy(), scale_x=micro, scale_y=micro)

    @pytest.fixture
    def small_scan_image(self) -> ScanImage:
        """Create a small scan image (<=20 columns) for testing."""
        data = np.ones((30, 15)) * 100.0
        return ScanImage(data=data, scale_x=micro, scale_y=micro)

    @pytest.fixture
    def full_mask(self) -> BinaryMask:
        """Create a full mask (all True)."""
        return np.ones((50, 50), dtype=bool)

    @pytest.fixture
    def partial_mask(self) -> BinaryMask:
        """Create a partial mask."""
        mask = np.ones((50, 50), dtype=bool)
        mask[10:20, 10:20] = False
        return mask

    def test_basic_functionality_no_needles(
        self, simple_scan_image: ScanImage, full_mask: BinaryMask
    ):
        """Test that data without needles remains mostly unchanged."""
        maskinator = Mask(mask=full_mask, remove_needles=True)
        result = maskinator(simple_scan_image).unwrap()

        # Result should have same shape
        assert result.data.shape == simple_scan_image.data.shape

        # Most values should be close to original (no outliers to remove)
        assert np.nanmean(np.abs(result.data - simple_scan_image.data)) < 5.0

    def test_removes_needles(self, simple_scan_image: ScanImage, full_mask: BinaryMask):
        """Test that obvious spike outliers are detected and set to NaN."""
        # Add some obvious spikes
        simple_scan_image.data[10, 10] = 1000.0  # Huge spike
        simple_scan_image.data[20, 20] = 1000.0
        simple_scan_image.data[30, 30] = 1000.0
        maskinator = Mask(mask=full_mask, remove_needles=True)

        result = maskinator.apply_on_image(simple_scan_image)

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
        scan_image = ScanImage(data=data, scale_x=micro, scale_y=micro)
        mask = np.ones((1, 50), dtype=bool)
        maskinator = Mask(mask=mask, remove_needles=True)

        result = maskinator.apply_on_image(scan_image)
        assert result.data.shape == (1, 50)
        # Needle should be detected
        assert np.isnan(result.data[0, 25])

    def test_all_nan_input(self):
        """Test handling of input data that is all NaN."""
        data = np.full((50, 50), np.nan)
        scan_image = ScanImage(data=data, scale_x=micro, scale_y=micro)
        mask = np.ones((50, 50), dtype=bool)
        maskinator = Mask(mask=mask, remove_needles=True)

        result = maskinator(scan_image).unwrap()

        # Should return all NaN
        assert np.all(np.isnan(result.data))

    def test_partial_nan_input(self, simple_scan_image, full_mask: BinaryMask):
        """Test handling of input with some NaN values."""
        simple_scan_image.data[5:10, 5:10] = np.nan
        simple_scan_image.data[30, 30] = 500.0  # Add needle
        maskinator = Mask(mask=full_mask, remove_needles=True)

        result = maskinator.apply_on_image(simple_scan_image)

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
        maskinator = Mask(mask=mask, remove_needles=True)

        result = maskinator(simple_scan_image).unwrap()

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
        maskinator = Mask(mask=full_mask, remove_needles=True)

        result = maskinator(scan_image).unwrap()

        # Count NaN values (should be minimal, just the outliers)
        nan_count = np.sum(np.isnan(result.data))
        total_count = result.data.size

        # Less than 5% should be NaN
        assert (nan_count / total_count) < 0.05

    def test_negative_outliers(self, simple_scan_image, full_mask):
        """Test detection of negative outliers (valleys)."""
        simple_scan_image.data[15, 15] = -500.0  # Deep valley
        simple_scan_image.data[35, 35] = -500.0

        maskinator = Mask(mask=full_mask, remove_needles=True)

        result = maskinator.apply_on_image(simple_scan_image)

        # Negative outliers should also be detected
        assert np.isnan(result.data[15, 15])
        assert np.isnan(result.data[35, 35])

    def test_no_modification_of_input(self, simple_scan_image, full_mask):
        """Test that input scan_image is not modified."""
        original_data = simple_scan_image.data.copy()

        maskinator = Mask(mask=full_mask, remove_needles=True)

        result = maskinator.apply_on_image(simple_scan_image)

        # Original should be unchanged
        assert_array_equal(simple_scan_image.data, original_data)
        # Result should be different object
        assert result.data is not simple_scan_image.data


class TestGetResidualImage(unittest.TestCase):
    """Tests for get_residual_image function."""

    def test_large_image_with_subsampling(self):
        """Test large image that requires subsampling will return same shape as input."""
        data = np.ones((100, 100)) * 10.0
        scan_image = ScanImage(data=data, scale_x=micro, scale_y=micro)
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        residual = maskinator.get_residual_image(scan_image)

        assert residual.shape == scan_image.data.shape

    def test_residual_detects_spikes(self):
        """Test that residuals are large for spike outliers."""
        data = np.ones((20, 20)) * 10.0
        data[10, 10] = 1000.0  # Add spike
        scan_image = ScanImage(data=data, scale_x=micro, scale_y=micro)

        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        residual = maskinator.get_residual_image(scan_image)

        # Spike location should have large residual
        assert np.abs(residual[10, 10]) > np.abs(residual[5, 5])

    def test_residual_image_of_scan_image_width_1(self):
        """Test that residuals are large for spike outliers."""
        data = np.ones((20, 1)) * 10.0
        data[10, 0] = 1000.0  # Add spike
        scan_image = ScanImage(data=data, scale_x=micro, scale_y=micro)

        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        residual = maskinator.get_residual_image(scan_image)

        # Spike location should have large residual
        assert np.abs(residual[10, 0]) > np.abs(residual[5, 0])
        assert residual.shape == scan_image.data.shape

    def test_residual_image_of_scan_image_height_1(self):
        """Test that residuals are large for spike outliers."""
        data = np.ones((1, 20)) * 10.0
        data[0, 10] = 1000.0  # Add spike
        scan_image = ScanImage(data=data, scale_x=micro, scale_y=micro)

        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        residual = maskinator.get_residual_image(scan_image)

        # Spike location should have large residual
        assert np.abs(residual[0, 10]) > np.abs(residual[0, 5])
        assert residual.shape == scan_image.data.shape

    def test_with_nan_values(self):
        """Test handling of NaN values in input."""
        data = np.ones((20, 20))
        data[5, 5] = np.nan
        scan_image = ScanImage(data=data, scale_x=micro, scale_y=micro)

        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        residual = maskinator.get_residual_image(scan_image)

        assert residual.shape == data.shape
        # NaN should propagate through
        assert np.isnan(residual[5, 5])


class TestApplyMedianFilter(unittest.TestCase):
    """Unit tests for apply_median_filter function."""

    def test_basic_3x3_filter(self):
        """Test basic 3x3 median filter on small array."""
        input_image = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
        )

        filter_size = 3
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)
        result = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)
        # Test with even filter_size (should become 5)
        result_even = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size=4
        )
        # Test with odd filter_size (should stay 5)
        result_odd = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # All pixels should be 10.0 since NaN padding is ignored
        assert result.data.shape == input_image.shape
        np.testing.assert_array_almost_equal(result.data, input_image, decimal=5)

    def test_large_filter_size(self):
        """Test with larger filter size."""
        input_image = np.random.rand(10, 10).astype(np.float64)

        filter_size = 7
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_filter_size_larger_than_image(self):
        """Test with larger filter size."""
        input_image = np.random.rand(10, 10).astype(np.float64)

        filter_size = 15
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_single_pixel_image(self):
        """Test edge case with 1x1 image."""
        input_image = np.array([[5.0]], dtype=np.float64)

        filter_size = 3
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == (1, 1)
        # With NaN padding, the single pixel should remain unchanged
        self.assertAlmostEqual(result.data[0, 0], 5.0, places=5)

    def test_rectangular_image(self):
        """Test with non-square image."""
        input_image = np.random.rand(5, 10).astype(np.float64)

        filter_size = 3
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_all_nan_input(self):
        """Test with all NaN input."""
        input_image = np.full((3, 3), np.nan, dtype=np.float64)

        filter_size = 3
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape

    def test_output_dtype_is_float64(self):
        """Test that output is converted to float64 (double in MATLAB)."""
        input_image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

        filter_size = 3
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        # MATLAB function converts to double at the end
        assert result.data.dtype == np.float64

    def test_large_image_performance(self):
        """Test with larger image to verify it handles size reasonably."""
        input_image = np.random.rand(100, 100).astype(np.float64)

        filter_size = 5
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
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
        maskinator = Mask(mask=np.zeros((2, 2), dtype=bool), remove_needles=True)

        result = maskinator.apply_median_filter(
            ScanImage(data=input_image, scale_x=1.0, scale_y=1.0), filter_size
        )

        assert result.data.shape == input_image.shape
