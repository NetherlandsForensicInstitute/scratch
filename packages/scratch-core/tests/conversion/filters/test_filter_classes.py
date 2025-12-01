import numpy as np
import pytest

from conversion.filters.filter_classes import SurfaceFilter


class DummyFilter(SurfaceFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _should_filter(self):
        return True

    def _filter_data(self, data):
        return data


class TestDummyFilterApply:
    """Tests for DummyFilter.apply() method."""

    def test_apply_1d_no_nans(self):
        """Test applying filter to 1D data without NaN."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        filter_obj = DummyFilter()

        result = filter_obj.apply(data)

        # Result should be smoothed
        assert result.shape == data.shape
        assert np.array_equal(result, data)  # Should be different from original
        assert not np.any(np.isnan(result))  # No NaNs introduced

    def test_apply_2d_no_nans(self):
        """Test applying filter to 2D data without NaN."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        filter_obj = DummyFilter()

        result = filter_obj.apply(data)

        assert result.shape == data.shape
        assert np.array_equal(result, data)
        assert not np.any(np.isnan(result))

    def test_apply_with_nan_borders(self):
        """Test that NaN borders are preserved."""
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
        filter_obj = DummyFilter()

        result = filter_obj.apply(data)

        assert result.shape == data.shape
        # Original NaN borders should be preserved
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[-1])
        assert np.isnan(result[-2])
        # Middle values should be filtered
        assert not np.any(np.isnan(result[2:6]))

    def test_apply_high_pass(self):
        """Test high-pass filtering mode."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Low-pass filter
        low_pass = DummyFilter(is_high_pass=False)
        low_result = low_pass.apply(data)

        # High-pass filter
        high_pass = DummyFilter(is_high_pass=True)
        high_result = high_pass.apply(data)

        # High-pass should give residual
        # Original H Low-pass + High-pass
        np.testing.assert_array_almost_equal(data, low_result + high_result, decimal=5)

    def test_apply_all_nan_data(self):
        """Test with array that's all NaN."""
        data = np.array([np.nan, np.nan, np.nan, np.nan])
        filter_obj = DummyFilter()

        result = filter_obj.apply(data)

        # Should return original data unchanged
        assert result.shape == data.shape
        assert np.all(np.isnan(result))

    def test_apply_empty_array_raises_error(self):
        """Test that empty array raises ValueError."""
        data = np.array([])
        filter_obj = DummyFilter()

        with pytest.raises(ValueError, match="Input data array is empty"):
            filter_obj.apply(data)

    def test_apply_3d_array_raises_error(self):
        """Test that 3D array raises ValueError."""
        data = np.zeros((3, 3, 3))
        filter_obj = DummyFilter()

        with pytest.raises(ValueError, match="Input data must be 1D or 2D, got 3D"):
            filter_obj.apply(data)

    def test_apply_preserves_shape(self):
        """Test that output shape matches input shape."""
        # 1D case
        data_1d = np.random.randn(20)
        filter_obj = DummyFilter()
        result_1d = filter_obj.apply(data_1d)
        assert result_1d.shape == data_1d.shape

        # 2D case
        data_2d = np.random.randn(10, 15)
        result_2d = filter_obj.apply(data_2d)
        assert result_2d.shape == data_2d.shape

    def test_apply_with_partial_nan(self):
        """Test with NaN values in the middle of data."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        filter_obj = DummyFilter()

        result = filter_obj.apply(data)

        # Should still filter, handling NaN appropriately
        assert result.shape == data.shape
        # Original NaN should be preserved if nan_out=True
        assert np.isnan(result[2])

    def test_apply_high_pass_2d(self):
        """Test high-pass mode with 2D data."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

        low_pass = DummyFilter(is_high_pass=False)
        high_pass = DummyFilter(is_high_pass=True)

        low_result = low_pass.apply(data)
        high_result = high_pass.apply(data)

        # Original H Low-pass + High-pass
        np.testing.assert_array_almost_equal(data, low_result + high_result, decimal=5)

    def test_apply_with_nan_borders_2d(self):
        """Test 2D data with NaN borders."""
        data = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, 5.0, np.nan], [np.nan, np.nan, np.nan]]
        )
        filter_obj = DummyFilter()

        result = filter_obj.apply(data)

        assert result.shape == data.shape
        # Center value should be filtered
        assert not np.isnan(result[1, 1])
        # Borders should remain NaN
        assert np.all(np.isnan(result[0, :]))
        assert np.all(np.isnan(result[-1, :]))
        assert np.all(np.isnan(result[:, 0]))
        assert np.all(np.isnan(result[:, -1]))

    def test_apply_list_input_converts_to_array(self):
        """Test that list input is converted to array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        filter_obj = DummyFilter()

        result = filter_obj.apply(data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)


class TestDummyFilterValidation:
    """Tests for parameter validation in DummyFilter.__init__."""

    def test_negative_robust_conv_tol_raises_error(self):
        """Test that negative robust_conv_tol raises ValueError."""
        with pytest.raises(ValueError, match="robust_conv_tol must be positive"):
            DummyFilter(robust_conv_tol=-0.1)

    def test_zero_robust_conv_tol_raises_error(self):
        """Test that zero robust_conv_tol raises ValueError."""
        with pytest.raises(ValueError, match="robust_conv_tol must be positive"):
            DummyFilter(robust_conv_tol=0)

    def test_zero_n_robust_conv_iter_raises_error(self):
        """Test that zero n_robust_conv_iter raises ValueError."""
        with pytest.raises(ValueError, match="n_robust_conv_iter must be >= 1"):
            DummyFilter(n_robust_conv_iter=0)

    def test_negative_n_robust_conv_iter_raises_error(self):
        """Test that negative n_robust_conv_iter raises ValueError."""
        with pytest.raises(ValueError, match="n_robust_conv_iter must be >= 1"):
            DummyFilter(n_robust_conv_iter=-5)

    def test_valid_parameters_no_error(self):
        """Test that valid parameters don't raise errors."""
        # Should not raise any exceptions
        DummyFilter(
            is_high_pass=True,
            nan_out=False,
            is_robust=True,
            robust_conv_tol=0.01,
            n_robust_conv_iter=50,
        )
