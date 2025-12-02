"""Tests for filter_kernel function."""

import numpy as np
import pytest

from conversion.filters.data_formats import FilterDomain
from conversion.filters.kernels import (
    create_footprint,
    create_averaging_kernel,
    filter_kernel,
)


class TestFilterKernel2D:
    """Tests for 2D kernel filtering."""

    def test_uniform_kernel_constant_data(self):
        """Test that uniform kernel on constant data returns same constant."""
        data = np.full((10, 10), 5.0)
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data)
        np.testing.assert_allclose(result, 5.0)

    def test_uniform_kernel_preserves_shape(self):
        """Test that filtering preserves data shape."""
        data = np.random.rand(20, 30)
        kernel = np.ones((5, 5)) / 25
        result = filter_kernel(kernel, None, data)
        assert result.shape == data.shape

    def test_identity_kernel(self):
        """Test that identity kernel returns original data."""
        data = np.random.rand(10, 10)
        kernel = np.array([[1.0]])
        result = filter_kernel(kernel, None, data)
        np.testing.assert_allclose(result, data)

    def test_smoothing_reduces_variance(self):
        """Test that smoothing reduces variance."""
        np.random.seed(42)
        data = np.random.rand(20, 20)
        kernel = np.ones((5, 5)) / 25
        result = filter_kernel(kernel, None, data)
        # Interior variance should be lower
        assert np.var(result[3:-3, 3:-3]) < np.var(data[3:-3, 3:-3])


class TestFilterKernelSeparable:
    """Tests for separable kernel filtering."""

    def test_separable_constant_data(self):
        """Test separable kernel on constant data."""
        data = np.full((10, 10), 5.0)
        kernel_col = np.ones((5, 1)) / 5
        kernel_row = np.ones((1, 5)) / 5
        result = filter_kernel(kernel_col, kernel_row, data)
        np.testing.assert_allclose(result, 5.0)

    def test_separable_preserves_shape(self):
        """Test that separable filtering preserves data shape."""
        data = np.random.rand(20, 30)
        kernel_col = np.ones((5, 1)) / 5
        kernel_row = np.ones((1, 7)) / 7
        result = filter_kernel(kernel_col, kernel_row, data)
        assert result.shape == data.shape

    def test_separable_equivalent_to_2d(self):
        """Test that separable filtering equals full 2D for uniform kernel."""
        np.random.seed(42)
        data = np.random.rand(20, 20)

        # Separable
        kernel_col = np.ones((5, 1)) / 5
        kernel_row = np.ones((1, 5)) / 5
        result_sep = filter_kernel(kernel_col, kernel_row, data)

        # Full 2D
        kernel_2d = np.ones((5, 5)) / 25
        result_2d = filter_kernel(kernel_2d, None, data)

        np.testing.assert_allclose(result_sep, result_2d)


class TestFilterKernelTuple:
    """Tests for tuple kernel input."""

    def test_tuple_kernel_constant_data(self):
        """Test tuple kernel on constant data."""
        data = np.full((10, 10), 5.0)
        kernel = (np.ones((5, 1)) / 5, np.ones((1, 5)) / 5)
        result = filter_kernel(kernel, None, data)
        np.testing.assert_allclose(result, 5.0)

    def test_tuple_kernel_matches_separate_args(self):
        """Test that tuple kernel matches separate col/row arguments."""
        np.random.seed(42)
        data = np.random.rand(20, 20)

        kernel_col = np.array([[0.1], [0.2], [0.4], [0.2], [0.1]])
        kernel_row = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])

        result_separate = filter_kernel(kernel_col, kernel_row, data)
        result_tuple = filter_kernel((kernel_col, kernel_row), None, data)

        np.testing.assert_allclose(result_separate, result_tuple)


class TestFilterKernelNaN:
    """Tests for NaN handling."""

    def test_nan_preserved_with_nan_out_true(self):
        """Test that NaN values are preserved when nan_out=True."""
        data = np.ones((10, 10))
        data[5, 5] = np.nan
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data, nan_out=True)
        assert np.isnan(result[5, 5])

    def test_nan_not_preserved_with_nan_out_false(self):
        """Test that NaN values are not preserved when nan_out=False."""
        data = np.ones((10, 10))
        data[5, 5] = np.nan
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data, nan_out=False)
        assert not np.isnan(result[5, 5])

    def test_nan_block_preserved(self):
        """Test that block of NaN values is preserved."""
        data = np.ones((20, 20))
        data[8:12, 8:12] = np.nan
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data)
        assert np.all(np.isnan(result[8:12, 8:12]))

    def test_nan_neighbors_handled(self):
        """Test that neighbors of NaN are still computed."""
        data = np.ones((10, 10))
        data[5, 5] = np.nan
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data)
        # Neighbors should have valid values
        assert not np.isnan(result[4, 5])
        assert not np.isnan(result[6, 5])


class TestFilterKernelWeights:
    """Tests for weighted filtering."""

    def test_uniform_weights_same_as_no_weights(self):
        """Test that uniform weights equal no weights."""
        np.random.seed(42)
        data = np.random.rand(20, 20)
        kernel = np.ones((5, 5)) / 25

        result_no_weights = filter_kernel(kernel, None, data)
        result_uniform_weights = filter_kernel(
            kernel, None, data, weights=np.ones_like(data)
        )

        np.testing.assert_allclose(result_no_weights, result_uniform_weights)

    def test_zero_weights_ignored(self):
        """Test that zero-weighted pixels are ignored."""
        data = np.ones((10, 10))
        data[5, 5] = 100.0  # Outlier
        weights = np.ones_like(data)
        weights[5, 5] = 0.0  # Ignore the outlier
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data, weights=weights)
        # Center should be close to 1.0, not affected by 100.0
        assert result[5, 5] < 2.0

    def test_partial_weights(self):
        """Test filtering with partial weights."""
        data = np.full((10, 10), 2.0)
        weights = np.ones_like(data)
        weights[5:, :] = 0.5  # Lower weights on bottom half
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data, weights=weights)
        # Result should still be close to 2.0 everywhere
        np.testing.assert_allclose(result, 2.0, atol=1e-10)


class TestFilterKernelRegressionOrder:
    """Tests for regression order parameter."""

    def test_regression_order_0(self):
        """Test that regression_order=0 works."""
        data = np.random.rand(10, 10)
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data, regression_order=0)
        assert result.shape == data.shape

    def test_regression_order_1_not_implemented(self):
        """Test that regression_order=1 raises NotImplementedError."""
        data = np.random.rand(10, 10)
        kernel = np.ones((3, 3)) / 9
        with pytest.raises(NotImplementedError):
            filter_kernel(kernel, None, data, regression_order=1)

    def test_regression_order_2_not_implemented(self):
        """Test that regression_order=2 raises NotImplementedError."""
        data = np.random.rand(10, 10)
        kernel = np.ones((3, 3)) / 9
        with pytest.raises(NotImplementedError):
            filter_kernel(kernel, None, data, regression_order=2)

    def test_invalid_regression_order(self):
        """Test that invalid regression_order raises ValueError."""
        data = np.random.rand(10, 10)
        kernel = np.ones((3, 3)) / 9
        with pytest.raises(ValueError):
            filter_kernel(kernel, None, data, regression_order=3)


class TestFilterKernelBoundary:
    """Tests for boundary handling."""

    def test_boundary_values_computed(self):
        """Test that boundary values are computed (not NaN)."""
        data = np.random.rand(10, 10)
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data)
        # Corners should have values
        assert not np.isnan(result[0, 0])
        assert not np.isnan(result[0, -1])
        assert not np.isnan(result[-1, 0])
        assert not np.isnan(result[-1, -1])

    def test_boundary_uses_available_data(self):
        """Test that boundary averaging uses only available data."""
        data = np.full((5, 5), 1.0)
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data)
        # All values should be 1.0 (averaging 1s with proper normalization)
        np.testing.assert_allclose(result, 1.0)


class TestFilterKernelEdgeCases:
    """Tests for edge cases."""

    def test_single_pixel_data(self):
        """Test filtering single pixel data."""
        data = np.array([[5.0]])
        kernel = np.array([[1.0]])
        result = filter_kernel(kernel, None, data)
        assert result[0, 0] == 5.0

    def test_all_nan_data(self):
        """Test filtering all-NaN data."""
        data = np.full((5, 5), np.nan)
        kernel = np.ones((3, 3)) / 9
        result = filter_kernel(kernel, None, data)
        assert np.all(np.isnan(result))

    def test_large_kernel(self):
        """Test with large kernel."""
        data = np.random.rand(50, 50)
        kernel = np.ones((15, 15)) / 225
        result = filter_kernel(kernel, None, data)
        assert result.shape == data.shape


class TestCreateKernelDomainRectangle:
    """Tests for rectangle domain."""

    def test_square_rectangle(self):
        """Test square rectangle kernel."""
        result = create_footprint(np.array([5, 5]), FilterDomain.RECTANGLE)
        assert result.shape == (5, 5)
        assert result.dtype == bool
        assert np.all(result)

    def test_non_square_rectangle(self):
        """Test non-square rectangle kernel."""
        result = create_footprint(np.array([3, 7]), FilterDomain.RECTANGLE)
        assert result.shape == (3, 7)
        assert np.all(result)

    def test_1d_row_rectangle(self):
        """Test 1D row kernel."""
        result = create_footprint(np.array([1, 5]), FilterDomain.RECTANGLE)
        assert result.shape == (1, 5)
        assert np.all(result)

    def test_1d_col_rectangle(self):
        """Test 1D column kernel."""
        result = create_footprint(np.array([5, 1]), FilterDomain.RECTANGLE)
        assert result.shape == (5, 1)
        assert np.all(result)


class TestCreateKernelDomainDisk:
    """Tests for disk domain."""

    def test_3x3_disk(self):
        """Test 3x3 disk kernel."""
        result = create_footprint(np.array([3, 3]), FilterDomain.DISK)
        expected = np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        )
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, expected)

    def test_5x5_disk(self):
        """Test 5x5 disk kernel."""
        result = create_footprint(np.array([5, 5]), FilterDomain.DISK)
        expected = np.array(
            [
                [False, True, True, True, False],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [False, True, True, True, False],
            ]
        )
        assert result.shape == (5, 5)
        np.testing.assert_array_equal(result, expected)

    def test_7x7_disk(self):
        """Test 7x7 disk kernel."""
        result = create_footprint(np.array([7, 7]), FilterDomain.DISK)
        expected = np.array(
            [
                [False, False, True, True, True, False, False],
                [False, True, True, True, True, True, False],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [False, True, True, True, True, True, False],
                [False, False, True, True, True, False, False],
            ]
        )
        assert result.shape == (7, 7)
        np.testing.assert_array_equal(result, expected)

    def test_ellipse_5x7(self):
        """Test 5x7 elliptical kernel."""
        result = create_footprint(np.array([5, 7]), FilterDomain.DISK)
        expected = np.array(
            [
                [False, True, True, True, True, True, False],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [False, True, True, True, True, True, False],
            ]
        )
        assert result.shape == (5, 7)
        np.testing.assert_array_equal(result, expected)

    def test_ellipse_7x5(self):
        """Test 7x5 elliptical kernel."""
        result = create_footprint(np.array([7, 5]), FilterDomain.DISK)
        expected = np.array(
            [
                [False, True, True, True, False],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [False, True, True, True, False],
            ]
        )
        assert result.shape == (7, 5)
        np.testing.assert_array_equal(result, expected)

    def test_disk_is_symmetric(self):
        """Test that disk kernels are symmetric."""
        for size in [(5, 5), (7, 7), (9, 9), (11, 11)]:
            result = create_footprint(np.array(size), FilterDomain.DISK)
            # Check horizontal symmetry
            np.testing.assert_array_equal(result, np.flip(result, axis=0))
            # Check vertical symmetry
            np.testing.assert_array_equal(result, np.flip(result, axis=1))

    def test_disk_center_is_always_true(self):
        """Test that the center pixel is always True."""
        for size in [(3, 3), (5, 5), (7, 7), (9, 9)]:
            result = create_footprint(np.array(size), FilterDomain.DISK)
            center = (size[0] // 2, size[1] // 2)
            assert result[center]


class TestCreateKernelDomain1DFallback:
    """Tests for 1D kernels falling back to rectangle."""

    def test_1d_row_with_disk_becomes_rectangle(self):
        """Test that 1D row kernel with DISK domain becomes rectangle."""
        result = create_footprint(np.array([1, 5]), FilterDomain.DISK)
        assert result.shape == (1, 5)
        assert np.all(result)

    def test_1d_col_with_disk_becomes_rectangle(self):
        """Test that 1D column kernel with DISK domain becomes rectangle."""
        result = create_footprint(np.array([5, 1]), FilterDomain.DISK)
        assert result.shape == (5, 1)
        assert np.all(result)


class TestCreateKernelDomainValidation:
    """Tests for input validation."""

    def test_invalid_domain_string(self):
        """Test that string domain raises ValueError."""
        with pytest.raises(ValueError, match="domain must be FilterDomain"):
            create_footprint(np.array([5, 5]), "disk")  # type: ignore[arg-type]

    def test_invalid_domain_none(self):
        """Test that None domain raises ValueError."""
        with pytest.raises(ValueError, match="domain must be FilterDomain"):
            create_footprint(np.array([5, 5]), None)  # type: ignore[arg-type]

    def test_invalid_size_zero(self):
        """Test that zero size raises ValueError."""
        with pytest.raises(ValueError, match="Kernel size must be >= 1"):
            create_footprint(np.array([0, 5]), FilterDomain.DISK)

    def test_invalid_size_negative(self):
        """Test that negative size raises ValueError."""
        with pytest.raises(ValueError, match="Kernel size must be >= 1"):
            create_footprint(np.array([-1, 5]), FilterDomain.DISK)


class TestCreateKernelDomainEdgeCases:
    """Tests for edge cases."""

    def test_minimum_size_1x1(self):
        """Test minimum 1x1 kernel."""
        result = create_footprint(np.array([1, 1]), FilterDomain.RECTANGLE)
        assert result.shape == (1, 1)
        assert result[0, 0]

    def test_large_kernel(self):
        """Test large kernel."""
        result = create_footprint(np.array([101, 101]), FilterDomain.DISK)
        assert result.shape == (101, 101)
        # Center should be True
        assert result[50, 50]
        # Corners should be False
        assert not result[0, 0]
        assert not result[0, 100]
        assert not result[100, 0]
        assert not result[100, 100]


class TestCreateAveragingKernelShape:
    """Tests for kernel shape."""

    def test_3x3_shape(self):
        """Test 3x3 kernel shape."""
        result = create_averaging_kernel(np.array([3, 3]))
        assert result.shape == (3, 3)

    def test_5x5_shape(self):
        """Test 5x5 kernel shape."""
        result = create_averaging_kernel(np.array([5, 5]))
        assert result.shape == (5, 5)

    def test_5x7_shape(self):
        """Test 5x7 kernel shape."""
        result = create_averaging_kernel(np.array([5, 7]))
        assert result.shape == (5, 7)

    def test_7x5_shape(self):
        """Test 7x5 kernel shape."""
        result = create_averaging_kernel(np.array([7, 5]))
        assert result.shape == (7, 5)


class TestCreateAveragingKernelNormalization:
    """Tests for kernel normalization (sum to 1)."""

    def test_3x3_sums_to_one(self):
        """Test 3x3 kernel sums to 1."""
        result = create_averaging_kernel(np.array([3, 3]))
        assert np.isclose(np.sum(result), 1.0)

    def test_5x5_sums_to_one(self):
        """Test 5x5 kernel sums to 1."""
        result = create_averaging_kernel(np.array([5, 5]))
        assert np.isclose(np.sum(result), 1.0)

    def test_7x7_sums_to_one(self):
        """Test 7x7 kernel sums to 1."""
        result = create_averaging_kernel(np.array([7, 7]))
        assert np.isclose(np.sum(result), 1.0)

    def test_ellipse_5x7_sums_to_one(self):
        """Test 5x7 elliptical kernel sums to 1."""
        result = create_averaging_kernel(np.array([5, 7]))
        assert np.isclose(np.sum(result), 1.0)

    def test_large_kernel_sums_to_one(self):
        """Test large kernel sums to 1."""
        result = create_averaging_kernel(np.array([51, 51]))
        assert np.isclose(np.sum(result), 1.0)


class TestCreateAveragingKernelValues:
    """Tests for kernel values."""

    def test_nonzero_values_are_equal(self):
        """Test that all nonzero values in kernel are equal."""
        result = create_averaging_kernel(np.array([5, 5]))
        nonzero_values = result[result > 0]
        assert np.allclose(nonzero_values, nonzero_values[0])

    def test_zero_outside_disk(self):
        """Test that values outside disk domain are zero."""
        size = np.array([5, 5])
        result = create_averaging_kernel(size)
        domain = create_footprint(size, FilterDomain.DISK)
        # Values outside domain should be zero
        np.testing.assert_array_equal(result[~domain], 0)

    def test_nonzero_inside_disk(self):
        """Test that values inside disk domain are nonzero."""
        size = np.array([5, 5])
        result = create_averaging_kernel(size)
        domain = create_footprint(size, FilterDomain.DISK)
        # Values inside domain should be positive
        assert np.all(result[domain] > 0)

    def test_value_equals_1_over_count(self):
        """Test that nonzero values equal 1/count."""
        size = np.array([5, 5])
        result = create_averaging_kernel(size)
        domain = create_footprint(size, FilterDomain.DISK)
        expected_value = 1.0 / np.sum(domain)
        np.testing.assert_allclose(result[domain], expected_value)


class TestCreateAveragingKernelSymmetry:
    """Tests for kernel symmetry."""

    def test_horizontal_symmetry(self):
        """Test horizontal symmetry."""
        result = create_averaging_kernel(np.array([7, 7]))
        np.testing.assert_array_equal(result, np.flip(result, axis=0))

    def test_vertical_symmetry(self):
        """Test vertical symmetry."""
        result = create_averaging_kernel(np.array([7, 7]))
        np.testing.assert_array_equal(result, np.flip(result, axis=1))

    def test_ellipse_horizontal_symmetry(self):
        """Test horizontal symmetry for elliptical kernel."""
        result = create_averaging_kernel(np.array([5, 9]))
        np.testing.assert_array_equal(result, np.flip(result, axis=0))

    def test_ellipse_vertical_symmetry(self):
        """Test vertical symmetry for elliptical kernel."""
        result = create_averaging_kernel(np.array([5, 9]))
        np.testing.assert_array_equal(result, np.flip(result, axis=1))


class TestCreateAveragingKernelDiskDomain:
    """Tests verifying disk domain is used."""

    def test_5x5_corners_are_zero(self):
        """Test that 5x5 kernel has zero corners (disk shape)."""
        result = create_averaging_kernel(np.array([5, 5]))
        assert result[0, 0] == 0
        assert result[0, 4] == 0
        assert result[4, 0] == 0
        assert result[4, 4] == 0

    def test_5x5_center_is_nonzero(self):
        """Test that 5x5 kernel has nonzero center."""
        result = create_averaging_kernel(np.array([5, 5]))
        assert result[2, 2] > 0

    def test_matches_disk_domain(self):
        """Test that kernel matches disk domain pattern."""
        for size in [(5, 5), (7, 7), (9, 9), (5, 7), (7, 5)]:
            size_arr = np.array(size)
            result = create_averaging_kernel(size_arr)
            domain = create_footprint(size_arr, FilterDomain.DISK)
            # Kernel should be nonzero exactly where domain is True
            np.testing.assert_array_equal(result > 0, domain)


class TestCreateAveragingKernelSpecificValues:
    """Tests for specific expected values."""

    def test_3x3_kernel(self):
        """Test specific values for 3x3 kernel."""
        result = create_averaging_kernel(np.array([3, 3]))
        # 3x3 disk is all True (9 pixels)
        expected_value = 1.0 / 9
        expected = np.full((3, 3), expected_value)
        np.testing.assert_allclose(result, expected)

    def test_5x5_kernel(self):
        """Test specific values for 5x5 kernel."""
        result = create_averaging_kernel(np.array([5, 5]))
        domain = create_footprint(np.array([5, 5]), FilterDomain.DISK)
        count = np.sum(domain)  # Should be 21 for 5x5 disk
        expected_value = 1.0 / count

        # Check nonzero values
        np.testing.assert_allclose(result[domain], expected_value)
        # Check zero values
        np.testing.assert_array_equal(result[~domain], 0)
