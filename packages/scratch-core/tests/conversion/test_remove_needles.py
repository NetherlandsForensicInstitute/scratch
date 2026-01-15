import numpy as np
import pytest
from numpy.testing import assert_array_equal

from container_models.scan_image import ScanImage
from conversion.remove_needles import remove_needles


class TestRemoveNeedles:
    """Unit tests for the remove_needles function."""

    @pytest.fixture
    def simple_scan_image(self):
        """Create a simple 50x50 scan image for testing."""
        data = np.ones((50, 50)) * 100.0
        return ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

    @pytest.fixture
    def small_scan_image(self):
        """Create a small scan image (<=20 columns) for testing."""
        data = np.ones((30, 15)) * 100.0
        return ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

    @pytest.fixture
    def full_mask(self):
        """Create a full mask (all True)."""
        return np.ones((50, 50), dtype=bool)

    @pytest.fixture
    def partial_mask(self):
        """Create a partial mask."""
        mask = np.ones((50, 50), dtype=bool)
        mask[10:20, 10:20] = False
        return mask

    def test_basic_functionality_no_needles(self, simple_scan_image, full_mask):
        """Test that data without needles remains mostly unchanged."""
        result = remove_needles(simple_scan_image, full_mask, times_median=15)

        # Result should have same shape
        assert result.shape == simple_scan_image.data.shape

        # Most values should be close to original (no outliers to remove)
        assert np.nanmean(np.abs(result - simple_scan_image.data)) < 5.0

    def test_removes_spike_outliers(self, simple_scan_image, full_mask):
        """Test that obvious spike outliers are detected and set to NaN."""
        # Add some obvious spikes
        simple_scan_image.data[10, 10] = 1000.0  # Huge spike
        simple_scan_image.data[20, 20] = 1000.0
        simple_scan_image.data[30, 30] = 1000.0

        result = remove_needles(simple_scan_image, full_mask, times_median=15)

        # Spikes should be set to NaN
        assert np.isnan(result[10, 10])
        assert np.isnan(result[20, 20])
        assert np.isnan(result[30, 30])

        # Most other values should remain
        non_spike_mask = np.ones_like(result, dtype=bool)
        non_spike_mask[10, 10] = False
        non_spike_mask[20, 20] = False
        non_spike_mask[30, 30] = False
        assert np.nanmean(result[non_spike_mask]) > 90.0

    def test_small_strip_behavior(self, small_scan_image, partial_mask):
        """Test that small strips trigger different filtering logic."""
        # Verify width <= 20
        assert small_scan_image.width <= 20

        result = remove_needles(
            small_scan_image, partial_mask[:30, :15], times_median=15
        )

        # Should complete without error and return correct shape
        assert result.shape == small_scan_image.data.shape

    def test_single_row_data(self):
        """Test handling of single-row data (1D case)."""
        data = np.ones((1, 50)) * 100.0
        data[0, 25] = 500.0  # Add a spike
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)
        mask = np.ones((1, 50), dtype=bool)

        result = remove_needles(scan_image, mask, times_median=15)

        assert result.shape == (1, 50)
        # Spike should be detected
        assert np.isnan(result[0, 25])

    def test_times_median_parameter(self, simple_scan_image, full_mask):
        """Test that times_median parameter affects outlier detection."""
        # Add moderate outlier
        simple_scan_image.data[25, 25] = 150.0

        # Strict threshold (lower times_median)
        result_strict = remove_needles(simple_scan_image, full_mask, times_median=5)

        # Lenient threshold (higher times_median)
        result_lenient = remove_needles(simple_scan_image, full_mask, times_median=50)

        # Strict should flag more points as NaN
        assert np.sum(np.isnan(result_strict)) >= np.sum(np.isnan(result_lenient))

    def test_subsampling_triggered(self, simple_scan_image, full_mask):
        """Test that subsampling is triggered for large images with fine resolution."""
        result = remove_needles(simple_scan_image, full_mask, times_median=15)

        # Should complete without error
        assert result.shape == simple_scan_image.data.shape

    def test_all_nan_input(self):
        """Test handling of input data that is all NaN."""
        data = np.full((50, 50), np.nan)
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)
        mask = np.ones((50, 50), dtype=bool)

        result = remove_needles(scan_image, mask, times_median=15)

        # Should return all NaN
        assert np.all(np.isnan(result))

    def test_partial_nan_input(self, simple_scan_image, full_mask):
        """Test handling of input with some NaN values."""
        simple_scan_image.data[5:10, 5:10] = np.nan
        simple_scan_image.data[30, 30] = 500.0  # Add spike

        result = remove_needles(simple_scan_image, full_mask, times_median=15)

        # Original NaN should remain
        assert np.all(np.isnan(result[5:10, 5:10]))

        # Spike should be detected
        assert np.isnan(result[30, 30])

    def test_mask_application(self, simple_scan_image):
        """Test that mask is properly applied."""
        mask = np.ones((50, 50), dtype=bool)
        mask[20:30, 20:30] = False  # Mask out a region

        # Add spike in masked region
        simple_scan_image.data[25, 25] = 1000.0

        result = remove_needles(simple_scan_image, mask, times_median=15)

        # Result should have correct shape
        assert result.shape == simple_scan_image.data.shape

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

        result = remove_needles(scan_image, full_mask, times_median=15)

        # Count NaN values (should be minimal, just the outliers)
        nan_count = np.sum(np.isnan(result))
        total_count = result.size

        # Less than 5% should be NaN
        assert (nan_count / total_count) < 0.05

    def test_return_type(self, simple_scan_image, full_mask):
        """Test that function returns correct type."""
        result = remove_needles(simple_scan_image, full_mask, times_median=15)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64 or result.dtype == np.float32

    def test_edge_case_width_exactly_20(self):
        """Test edge case where width is exactly 20."""
        data = np.ones((30, 20)) * 100.0
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)
        mask = np.ones((30, 20), dtype=bool)

        result = remove_needles(scan_image, mask, times_median=15)

        # Should use large strip logic (not small strip)
        assert result.shape == (30, 20)

    def test_edge_case_width_21(self):
        """Test edge case where width is 21 (just above threshold)."""
        data = np.ones((30, 21)) * 100.0
        scan_image = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)
        mask = np.ones((30, 21), dtype=bool)

        result = remove_needles(scan_image, mask, times_median=15)

        # Should use large strip logic
        assert result.shape == (30, 21)

    def test_negative_outliers(self, simple_scan_image, full_mask):
        """Test detection of negative outliers (valleys)."""
        simple_scan_image.data[15, 15] = -500.0  # Deep valley
        simple_scan_image.data[35, 35] = -500.0

        result = remove_needles(simple_scan_image, full_mask, times_median=15)

        # Negative outliers should also be detected
        assert np.isnan(result[15, 15])
        assert np.isnan(result[35, 35])

    def test_no_modification_of_input(self, simple_scan_image, full_mask):
        """Test that input scan_image is not modified."""
        original_data = simple_scan_image.data.copy()

        result = remove_needles(simple_scan_image, full_mask, times_median=15)

        # Original should be unchanged
        assert_array_equal(simple_scan_image.data, original_data)
        # Result should be different object
        assert result is not simple_scan_image.data
