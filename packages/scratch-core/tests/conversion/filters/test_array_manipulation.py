import numpy as np
import pytest

from conversion.filters.array_manipulation import (
    crop_array,
    pad_array,
    calculate_nan_trim,
)
from conversion.filters.data_formats import Trim1D, Trim2D, Pad1D, Pad2D


class TestCalculateNanTrim1D:
    """Tests for 1D arrays."""

    def test_no_nans(self):
        """Test array with no NaN values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=0, end=0)
        np.testing.assert_array_equal(result, expected)

    def test_nan_at_start(self):
        """Test array with NaN at the start."""
        data = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=2, end=0)
        np.testing.assert_array_equal(result, expected)

    def test_nan_at_end(self):
        """Test array with NaN at the end."""
        data = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=0, end=2)
        np.testing.assert_array_equal(result, expected)

    def test_nan_both_ends(self):
        """Test array with NaN at both ends."""
        data = np.array([np.nan, np.nan, 3.0, 4.0, np.nan])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=2, end=1)
        np.testing.assert_array_equal(result, expected)

    def test_all_nan(self):
        """Test array with all NaN values."""
        data = np.array([np.nan, np.nan, np.nan])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=0, end=3)
        np.testing.assert_array_equal(result, expected)

    def test_single_valid_value(self):
        """Test array with single valid value."""
        data = np.array([np.nan, 5.0, np.nan, np.nan])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=1, end=2)
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test empty array."""
        data = np.array([])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=0, end=0)
        np.testing.assert_array_equal(result, expected)


class TestCalculateNanTrim2D:
    """Tests for 2D arrays."""

    def test_no_nans(self):
        """Test 2D array with no NaN values."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = calculate_nan_trim(data)
        expected = Trim2D(top=0, bottom=0, left=0, right=0)
        np.testing.assert_array_equal(result, expected)

    def test_nan_border_all_sides(self):
        """Test 2D array with NaN border on all sides."""
        data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 5.0, 3.0, np.nan],
                [np.nan, 2.0, 7.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=1, bottom=1, left=1, right=1)
        np.testing.assert_array_equal(result, expected)

    def test_nan_top_and_left(self):
        """Test 2D array with NaN on top and left."""
        data = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, 5.0, 3.0], [np.nan, 2.0, 7.0]]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=1, bottom=0, left=1, right=0)
        np.testing.assert_array_equal(result, expected)

    def test_nan_bottom_and_right(self):
        """Test 2D array with NaN on bottom and right."""
        data = np.array(
            [[1.0, 2.0, np.nan], [3.0, 4.0, np.nan], [np.nan, np.nan, np.nan]]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=0, bottom=1, left=0, right=1)
        np.testing.assert_array_equal(result, expected)

    def test_all_nan(self):
        """Test 2D array with all NaN values."""
        data = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=0, bottom=3, left=0, right=3)
        np.testing.assert_array_equal(result, expected)

    def test_single_valid_value(self):
        """Test 2D array with single valid value."""
        data = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, 5.0, np.nan], [np.nan, np.nan, np.nan]]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=1, bottom=1, left=1, right=1)
        np.testing.assert_array_equal(result, expected)

    def test_valid_row_in_middle(self):
        """Test 2D array with valid data only in middle row."""
        data = np.array(
            [[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=1, bottom=1, left=0, right=0)
        np.testing.assert_array_equal(result, expected)

    def test_valid_column_in_middle(self):
        """Test 2D array with valid data only in middle column."""
        data = np.array(
            [[np.nan, 1.0, np.nan], [np.nan, 2.0, np.nan], [np.nan, 3.0, np.nan]]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=0, bottom=0, left=1, right=1)
        np.testing.assert_array_equal(result, expected)

    def test_asymmetric_borders(self):
        """Test 2D array with asymmetric NaN borders."""
        data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1.0, 2.0, 3.0, np.nan],
                [np.nan, 4.0, 5.0, 6.0, np.nan],
            ]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=2, bottom=0, left=1, right=1)
        np.testing.assert_array_equal(result, expected)

    def test_rectangular_array(self):
        """Test non-square 2D array."""
        data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1.0, 2.0, 3.0, 4.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ]
        )
        result = calculate_nan_trim(data)
        expected = Trim2D(top=1, bottom=1, left=1, right=1)
        np.testing.assert_array_equal(result, expected)

    def test_empty_2d_array(self):
        """Test empty 2D array."""
        data = np.array([]).reshape(0, 0)
        result = calculate_nan_trim(data)
        expected = Trim2D(top=0, bottom=0, left=0, right=0)
        np.testing.assert_array_equal(result, expected)


class TestCalculateNanTrimEdgeCases:
    """Tests for edge cases and error handling."""

    def test_3d_array_raises_error(self):
        """Test that 3D array raises ValueError."""
        data = np.zeros((3, 3, 3))
        with pytest.raises(ValueError, match="Data must be 1D or 2D, got 3D"):
            calculate_nan_trim(data)

    def test_4d_array_raises_error(self):
        """Test that 4D array raises ValueError."""
        data = np.zeros((2, 2, 2, 2))
        with pytest.raises(ValueError, match="Data must be 1D or 2D, got 4D"):
            calculate_nan_trim(data)

    def test_list_input_converts_to_array(self):
        """Test that list input is converted to array."""
        data = [np.nan, 1.0, 2.0, np.nan]
        result = calculate_nan_trim(np.array(data))
        expected = Trim1D(start=1, end=1)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_inf_and_nan(self):
        """Test array with both inf and NaN."""
        data = np.array([np.nan, np.inf, 2.0, np.nan])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=1, end=1)  # inf is valid, only NaN trimmed
        np.testing.assert_array_equal(result, expected)

    def test_negative_values(self):
        """Test array with negative values."""
        data = np.array([np.nan, -5.0, -3.0, np.nan])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=1, end=1)
        np.testing.assert_array_equal(result, expected)

    def test_zero_values(self):
        """Test array with zero values."""
        data = np.array([np.nan, 0.0, 0.0, np.nan])
        result = calculate_nan_trim(data)
        expected = Trim1D(start=1, end=1)
        np.testing.assert_array_equal(result, expected)


class TestCropArray1D:
    """Tests for crop_array with 1D arrays."""

    def test_no_crop(self):
        """Test cropping with zero trim."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trim = Trim1D(start=0, end=0)
        result = crop_array(data, trim)
        np.testing.assert_array_equal(result, data)

    def test_crop_start(self):
        """Test cropping from start only."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trim = Trim1D(start=2, end=0)
        result = crop_array(data, trim)
        expected = np.array([3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_crop_end(self):
        """Test cropping from end only."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trim = Trim1D(start=0, end=2)
        result = crop_array(data, trim)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_crop_both_ends(self):
        """Test cropping from both ends."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trim = Trim1D(start=1, end=1)
        result = crop_array(data, trim)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result, expected)

    def test_crop_to_single_element(self):
        """Test cropping to single element."""
        data = np.array([1.0, 2.0, 3.0])
        trim = Trim1D(start=1, end=1)
        result = crop_array(data, trim)
        expected = np.array([2.0])
        np.testing.assert_array_equal(result, expected)

    def test_crop_to_empty(self):
        """Test cropping to empty array."""
        data = np.array([1.0, 2.0, 3.0])
        trim = Trim1D(start=2, end=2)
        result = crop_array(data, trim)
        assert result.size == 0

    def test_crop_with_nan(self):
        """Test cropping array containing NaN."""
        data = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        trim = Trim1D(start=1, end=1)
        result = crop_array(data, trim)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_wrong_trim_type_raises_error(self):
        """Test that wrong trim type raises error."""
        data = np.array([1.0, 2.0, 3.0])
        trim = Trim2D(top=0, bottom=0, left=0, right=0)
        with pytest.raises(ValueError, match="Expected Trim1D"):
            crop_array(data, trim)


class TestCropArray2D:
    """Tests for crop_array with 2D arrays."""

    def test_no_crop(self):
        """Test cropping with zero trim."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        trim = Trim2D(top=0, bottom=0, left=0, right=0)
        result = crop_array(data, trim)
        np.testing.assert_array_equal(result, data)

    def test_crop_top(self):
        """Test cropping from top."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        trim = Trim2D(top=1, bottom=0, left=0, right=0)
        result = crop_array(data, trim)
        expected = np.array([[4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(result, expected)

    def test_crop_bottom(self):
        """Test cropping from bottom."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        trim = Trim2D(top=0, bottom=1, left=0, right=0)
        result = crop_array(data, trim)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_crop_left(self):
        """Test cropping from left."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        trim = Trim2D(top=0, bottom=0, left=1, right=0)
        result = crop_array(data, trim)
        expected = np.array([[2, 3], [5, 6], [8, 9]])
        np.testing.assert_array_equal(result, expected)

    def test_crop_right(self):
        """Test cropping from right."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        trim = Trim2D(top=0, bottom=0, left=0, right=1)
        result = crop_array(data, trim)
        expected = np.array([[1, 2], [4, 5], [7, 8]])
        np.testing.assert_array_equal(result, expected)

    def test_crop_all_sides(self):
        """Test cropping from all sides."""
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        trim = Trim2D(top=1, bottom=1, left=1, right=1)
        result = crop_array(data, trim)
        expected = np.array([[6, 7], [10, 11]])
        np.testing.assert_array_equal(result, expected)

    def test_crop_to_single_element(self):
        """Test cropping to single element."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        trim = Trim2D(top=1, bottom=1, left=1, right=1)
        result = crop_array(data, trim)
        expected = np.array([[5]])
        np.testing.assert_array_equal(result, expected)

    def test_crop_to_empty(self):
        """Test cropping to empty array."""
        data = np.array([[1, 2], [3, 4]])
        trim = Trim2D(top=1, bottom=1, left=1, right=1)
        result = crop_array(data, trim)
        assert result.size == 0

    def test_crop_rectangular_array(self):
        """Test cropping non-square array."""
        data = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        trim = Trim2D(top=0, bottom=1, left=1, right=1)
        result = crop_array(data, trim)
        expected = np.array([[2, 3, 4, 5]])
        np.testing.assert_array_equal(result, expected)

    def test_wrong_trim_type_raises_error(self):
        """Test that wrong trim type raises error."""
        data = np.array([[1, 2], [3, 4]])
        trim = Trim1D(start=0, end=0)
        with pytest.raises(ValueError, match="Expected Trim2D"):
            crop_array(data, trim)


class TestCropArray3D:
    """Tests for crop_array error handling with 3D arrays."""

    def test_3d_array_raises_error(self):
        """Test that 3D array raises error."""
        data = np.zeros((3, 3, 3))
        trim = Trim2D(top=0, bottom=0, left=0, right=0)
        with pytest.raises(ValueError, match="Data must be 1D or 2D, got 3D"):
            crop_array(data, trim)


class TestPadArray1D:
    """Tests for pad_array with 1D arrays."""

    def test_no_pad(self):
        """Test padding with zero pad."""
        data = np.array([1.0, 2.0, 3.0])
        pad = Pad1D(start=0, end=0)
        result = pad_array(data, pad)
        np.testing.assert_array_equal(result, data)

    def test_pad_start(self):
        """Test padding at start only."""
        data = np.array([1.0, 2.0, 3.0])
        pad = Pad1D(start=2, end=0)
        result = pad_array(data, pad)
        expected = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_pad_end(self):
        """Test padding at end only."""
        data = np.array([1.0, 2.0, 3.0])
        pad = Pad1D(start=0, end=2)
        result = pad_array(data, pad)
        expected = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_pad_both_ends(self):
        """Test padding at both ends."""
        data = np.array([1.0, 2.0, 3.0])
        pad = Pad1D(start=1, end=1)
        result = pad_array(data, pad)
        expected = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_pad_with_existing_nan(self):
        """Test padding array that already contains NaN."""
        data = np.array([1.0, np.nan, 3.0])
        pad = Pad1D(start=1, end=1)
        result = pad_array(data, pad)
        expected = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_pad_single_element(self):
        """Test padding single element array."""
        data = np.array([5.0])
        pad = Pad1D(start=2, end=2)
        result = pad_array(data, pad)
        expected = np.array([np.nan, np.nan, 5.0, np.nan, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_pad_empty_array(self):
        """Test padding empty array."""
        data = np.array([])
        pad = Pad1D(start=2, end=2)
        result = pad_array(data, pad)
        expected = np.array([np.nan, np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_wrong_pad_type_raises_error(self):
        """Test that wrong pad type raises error."""
        data = np.array([1.0, 2.0, 3.0])
        pad = Pad2D(top=0, bottom=0, left=0, right=0)
        with pytest.raises(ValueError, match="Expected Pad1D"):
            pad_array(data, pad)


class TestPadArray2D:
    """Tests for pad_array with 2D arrays."""

    def test_no_pad(self):
        """Test padding with zero pad."""
        data = np.array([[1, 2], [3, 4]])
        pad = Pad2D(top=0, bottom=0, left=0, right=0)
        result = pad_array(data, pad)
        np.testing.assert_array_equal(result, data.astype(float))

    def test_pad_top(self):
        """Test padding at top."""
        data = np.array([[1, 2], [3, 4]])
        pad = Pad2D(top=1, bottom=0, left=0, right=0)
        result = pad_array(data, pad)
        expected = np.array([[np.nan, np.nan], [1, 2], [3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_pad_bottom(self):
        """Test padding at bottom."""
        data = np.array([[1, 2], [3, 4]])
        pad = Pad2D(top=0, bottom=1, left=0, right=0)
        result = pad_array(data, pad)
        expected = np.array([[1, 2], [3, 4], [np.nan, np.nan]])
        np.testing.assert_array_equal(result, expected)

    def test_pad_left(self):
        """Test padding at left."""
        data = np.array([[1, 2], [3, 4]])
        pad = Pad2D(top=0, bottom=0, left=1, right=0)
        result = pad_array(data, pad)
        expected = np.array([[np.nan, 1, 2], [np.nan, 3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_pad_right(self):
        """Test padding at right."""
        data = np.array([[1, 2], [3, 4]])
        pad = Pad2D(top=0, bottom=0, left=0, right=1)
        result = pad_array(data, pad)
        expected = np.array([[1, 2, np.nan], [3, 4, np.nan]])
        np.testing.assert_array_equal(result, expected)

    def test_pad_all_sides(self):
        """Test padding on all sides."""
        data = np.array([[5]])
        pad = Pad2D(top=1, bottom=1, left=1, right=1)
        result = pad_array(data, pad)
        expected = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, 5, np.nan], [np.nan, np.nan, np.nan]]
        )
        np.testing.assert_array_equal(result, expected)

    def test_pad_rectangular_array(self):
        """Test padding non-square array."""
        data = np.array([[1, 2, 3]])
        pad = Pad2D(top=1, bottom=1, left=0, right=0)
        result = pad_array(data, pad)
        expected = np.array(
            [[np.nan, np.nan, np.nan], [1, 2, 3], [np.nan, np.nan, np.nan]]
        )
        np.testing.assert_array_equal(result, expected)

    def test_pad_with_existing_nan(self):
        """Test padding array with existing NaN."""
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        pad = Pad2D(top=1, bottom=0, left=0, right=1)
        result = pad_array(data, pad)
        expected = np.array(
            [[np.nan, np.nan, np.nan], [1, np.nan, np.nan], [3, 4, np.nan]]
        )
        np.testing.assert_array_equal(result, expected)

    def test_wrong_pad_type_raises_error(self):
        """Test that wrong pad type raises error."""
        data = np.array([[1, 2], [3, 4]])
        pad = Pad1D(start=0, end=0)
        with pytest.raises(ValueError, match="Expected Pad2D"):
            pad_array(data, pad)


class TestPadArray3D:
    """Tests for pad_array error handling with 3D arrays."""

    def test_3d_array_raises_error(self):
        """Test that 3D array raises error."""
        data = np.zeros((3, 3, 3))
        pad = Pad2D(top=0, bottom=0, left=0, right=0)
        with pytest.raises(ValueError, match="Data must be 1D or 2D, got 3D"):
            pad_array(data, pad)


class TestCropPadRoundTrip:
    """Tests for crop and pad working together."""

    def test_1d_roundtrip(self):
        """Test that crop then pad restores original size."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trim = Trim1D(start=1, end=1)

        cropped = crop_array(data, trim)
        restored = pad_array(cropped, trim.to_pad())

        assert restored.shape == data.shape
        # Middle values should be preserved
        np.testing.assert_array_equal(restored[1:4], data[1:4])
        # Borders should be NaN
        assert np.isnan(restored[0])
        assert np.isnan(restored[4])

    def test_2d_roundtrip(self):
        """Test that crop then pad restores original size."""
        data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
        trim = Trim2D(top=1, bottom=1, left=1, right=1)

        cropped = crop_array(data, trim)
        restored = pad_array(cropped, trim.to_pad())

        assert restored.shape == data.shape
        # Center values should be preserved
        np.testing.assert_array_equal(restored[1:3, 1:3], data[1:3, 1:3])
        # Borders should be NaN
        assert np.all(np.isnan(restored[0, :]))
        assert np.all(np.isnan(restored[-1, :]))
        assert np.all(np.isnan(restored[:, 0]))
        assert np.all(np.isnan(restored[:, -1]))
