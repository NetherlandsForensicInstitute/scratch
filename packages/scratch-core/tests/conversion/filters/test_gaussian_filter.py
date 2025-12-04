"""Tests for Gaussian filter - verifies MATLAB compatibility."""

import numpy as np
import pytest
from scipy import ndimage

from conversion.filters.filter_classes import GaussianFilter
from conversion.filters.gaussian_filter import (
    apply_gaussian_filter,
    _cutoff_to_sigma,
    _cutoff_to_truncate,
    get_alpha,
)


def matlab_kernel_gauss(cutoff):
    """Create Gaussian kernel matching MATLAB kernel_gauss.m."""
    alpha = np.sqrt(np.log(2) / np.pi)
    size = int(2 * np.ceil(cutoff) + 1)
    radius = (size - 1) // 2
    pos = np.arange(-radius, radius + 1)
    kernel_1d = (1 / (alpha * cutoff)) * np.exp(-np.pi * (pos / (alpha * cutoff)) ** 2)
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel / np.sum(kernel)


def matlab_filter(data, cutoff):
    """MATLAB filter_kernel.m reference implementation."""
    kernel = matlab_kernel_gauss(cutoff)

    weights = (~np.isnan(data)).astype(float)
    data_clean = np.where(np.isnan(data), 0, data)

    filtered = ndimage.convolve(data_clean, kernel, mode="constant", cval=0.0)
    weight_sum = ndimage.convolve(weights, kernel, mode="constant", cval=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = filtered / weight_sum
    result[np.isnan(data)] = np.nan
    return result


class TestCutoffToSigma:
    """Test cutoff to sigma conversion."""

    def test_conversion_formula(self):
        """Sigma = alpha * cutoff / sqrt(2*pi)."""
        cutoff = 5.0
        alpha = get_alpha(n_order=0)
        expected = alpha * cutoff / np.sqrt(2 * np.pi)
        assert _cutoff_to_sigma(alpha, cutoff) == pytest.approx(expected, rel=1e-15)

    def test_linear_scaling(self):
        """Sigma should scale linearly with cutoff."""
        sigma1 = _cutoff_to_sigma(get_alpha(n_order=0), 5.0)
        sigma2 = _cutoff_to_sigma(get_alpha(n_order=0), 10.0)
        assert sigma2 == pytest.approx(2 * sigma1, rel=1e-15)

    def test_various_cutoffs(self):
        """Test sigma calculation for various cutoffs."""
        for cutoff in [1, 3, 5, 7, 10, 20]:
            sigma = _cutoff_to_sigma(get_alpha(n_order=0), cutoff)
            assert sigma > 0
            assert sigma < cutoff  # sigma is always smaller than cutoff


class TestCutoffToTruncate:
    """Test cutoff to truncate conversion."""

    def test_conversion_formula(self):
        """Truncate = ceil(cutoff) / sigma."""
        cutoff = 5.0
        sigma = _cutoff_to_sigma(get_alpha(n_order=0), cutoff)
        expected = np.ceil(cutoff) / sigma
        assert _cutoff_to_truncate(cutoff, sigma) == pytest.approx(expected, rel=1e-15)

    def test_ensures_kernel_size(self):
        """Truncate should ensure kernel radius >= ceil(cutoff)."""
        for cutoff in [3, 5, 7, 10]:
            sigma = _cutoff_to_sigma(get_alpha(n_order=0), cutoff)
            truncate = _cutoff_to_truncate(cutoff, sigma)
            # scipy kernel radius = ceil(truncate * sigma)
            kernel_radius = np.ceil(truncate * sigma)
            assert kernel_radius >= np.ceil(cutoff)


class TestGaussianFilterFunction:
    """Test the gaussian_filter function."""

    def test_matches_matlab_basic(self):
        """Basic filtering should match MATLAB."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100
        cutoff = 5.0

        matlab_result = matlab_filter(data, cutoff)
        our_result = apply_gaussian_filter(data, (cutoff, cutoff))

        np.testing.assert_allclose(our_result, matlab_result, rtol=1e-10)

    def test_matches_matlab_various_cutoffs(self):
        """Should match MATLAB for various cutoff lengths."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100

        for cutoff in [3, 5, 7, 10, 15]:
            matlab_result = matlab_filter(data, cutoff)
            our_result = apply_gaussian_filter(data, (cutoff, cutoff))
            np.testing.assert_allclose(
                our_result,
                matlab_result,
                rtol=1e-10,
                err_msg=f"Failed for cutoff={cutoff}",
            )

    def test_with_nan_values(self):
        """Should handle NaN values correctly."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100
        data[5, 5] = np.nan
        data[15, 10] = np.nan
        data[20, 25] = np.nan

        matlab_result = matlab_filter(data, 5.0)
        our_result = apply_gaussian_filter(data, (5.0, 5.0))

        # NaN positions should match
        np.testing.assert_array_equal(np.isnan(our_result), np.isnan(matlab_result))

        # Non-NaN values should match
        mask = ~np.isnan(matlab_result)
        np.testing.assert_allclose(our_result[mask], matlab_result[mask], rtol=1e-10)

    def test_nan_out_false(self):
        """When nan_out=False, NaN positions get interpolated values."""
        data = np.ones((10, 10)) * 5.0
        data[5, 5] = np.nan

        result = apply_gaussian_filter(data, (3.0, 3.0), nan_out=False)

        # The NaN position should now have a value (interpolated from neighbors)
        assert not np.isnan(result[5, 5])
        assert result[5, 5] == pytest.approx(5.0, rel=0.1)

    def test_pixel_separation(self):
        """Pixel separation should scale cutoff correctly."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100

        # cutoff=5 with pixel_sep=0.5 should equal cutoff_pixels=10
        matlab_result = matlab_filter(data, 10.0)
        our_result = apply_gaussian_filter(data, (5.0, 5.0), pixel_size=(0.5, 0.5))

        np.testing.assert_allclose(our_result, matlab_result, rtol=1e-10)

    def test_asymmetric_pixel_separation(self):
        """Should handle different pixel separation in each direction."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100

        # Different separation in row vs col
        result = apply_gaussian_filter(data, (5.0, 5.0), pixel_size=(1.0, 0.5))

        assert result.shape == data.shape
        assert not np.any(np.isnan(result))

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        for shape in [(10, 10), (20, 30), (50, 25)]:
            data = np.random.rand(*shape)
            result = apply_gaussian_filter(data, (5.0, 5.0))
            assert result.shape == shape

    def test_uniform_data_unchanged(self):
        """Filtering uniform data should return (approximately) same values."""
        data = np.ones((20, 20)) * 42.0
        result = apply_gaussian_filter(data, (5.0, 5.0))

        # Interior should be exactly 42 (edges affected by zero-padding)
        np.testing.assert_allclose(result[5:15, 5:15], 42.0, rtol=1e-10)

    def test_smoothing_effect(self):
        """Filtering should reduce variance (smoothing)."""
        np.random.seed(42)
        data = np.random.rand(50, 50) * 100
        result = apply_gaussian_filter(data, (5.0, 5.0))

        # Variance should decrease after smoothing
        assert np.var(result) < np.var(data)


class TestGaussianFilterClass:
    """Test the GaussianFilter class."""

    def test_apply_matches_function(self):
        """Class apply() should match gaussian_filter function."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100

        f = GaussianFilter(cutoff_length=(5.0, 5.0))
        class_result = f.apply(data)
        func_result = apply_gaussian_filter(data, (5.0, 5.0))

        np.testing.assert_array_equal(class_result, func_result)

    def test_apply_matches_matlab(self):
        """Class apply() should match MATLAB."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100

        f = GaussianFilter(cutoff_length=(5.0, 5.0))
        result = f.apply(data)

        matlab_result = matlab_filter(data, 5.0)
        np.testing.assert_allclose(result, matlab_result, rtol=1e-10)

    def test_high_pass_mode(self):
        """High-pass should return data - lowpass."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100

        f_lp = GaussianFilter(cutoff_length=(5.0, 5.0), is_high_pass=False)
        f_hp = GaussianFilter(cutoff_length=(5.0, 5.0), is_high_pass=True)

        lowpass = f_lp.apply(data)
        highpass = f_hp.apply(data)

        np.testing.assert_allclose(highpass, data - lowpass, rtol=1e-15)

    def test_high_pass_sum_equals_original(self):
        """Lowpass + highpass should equal original data."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100

        f_lp = GaussianFilter(cutoff_length=(5.0, 5.0), is_high_pass=False)
        f_hp = GaussianFilter(cutoff_length=(5.0, 5.0), is_high_pass=True)

        lowpass = f_lp.apply(data)
        highpass = f_hp.apply(data)

        np.testing.assert_allclose(lowpass + highpass, data, rtol=1e-14)

    def test_nan_cutoff_returns_unchanged(self):
        """NaN cutoff should return data unchanged."""
        data = np.random.rand(10, 10)

        f = GaussianFilter(cutoff_length=(np.nan, np.nan))
        result = f.apply(data)

        np.testing.assert_array_equal(result, data)

    def test_pixel_separation(self):
        """Pixel separation should work correctly."""
        np.random.seed(42)
        data = np.random.rand(30, 30) * 100

        f = GaussianFilter(cutoff_length=(5.0, 5.0), pixel_size=(0.5, 0.5))
        result = f.apply(data)

        matlab_result = matlab_filter(data, 10.0)  # 5.0 / 0.5 = 10 pixels
        np.testing.assert_allclose(result, matlab_result, rtol=1e-10)

    def test_attributes_stored(self):
        """Constructor should store all attributes."""
        f = GaussianFilter(
            cutoff_length=(5.0, 7.0),
            pixel_size=(0.5, 1.0),
            is_high_pass=True,
            nan_out=False,
        )

        assert f.cutoff_length == (5.0, 7.0)
        assert f.pixel_size == (0.5, 1.0)
        assert f.is_high_pass is True
        assert f.nan_out is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_pixel(self):
        """Single pixel data should work."""
        data = np.array([[42.0]])
        result = apply_gaussian_filter(data, (3.0, 3.0))

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(42.0)

    def test_small_data_large_kernel(self):
        """Data smaller than kernel should work."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        cutoff = 10.0

        matlab_result = matlab_filter(data, cutoff)
        our_result = apply_gaussian_filter(data, (cutoff, cutoff))

        np.testing.assert_allclose(our_result, matlab_result, rtol=1e-10)

    def test_all_nan_data(self):
        """All-NaN data should return all NaN."""
        data = np.full((10, 10), np.nan)
        result = apply_gaussian_filter(data, (5.0, 5.0))

        assert np.all(np.isnan(result))

    def test_single_valid_value(self):
        """Single non-NaN value should be preserved."""
        data = np.full((10, 10), np.nan)
        data[5, 5] = 42.0

        result = apply_gaussian_filter(data, (3.0, 3.0))

        assert result[5, 5] == pytest.approx(42.0)

    def test_row_of_nan(self):
        """Row of NaN should be handled correctly."""
        np.random.seed(42)
        data = np.random.rand(20, 20) * 100
        data[10, :] = np.nan

        result = apply_gaussian_filter(data, (5.0, 5.0))

        # NaN row should remain NaN
        assert np.all(np.isnan(result[10, :]))
        # Other rows should have values
        assert not np.all(np.isnan(result[5, :]))

    def test_column_of_nan(self):
        """Column of NaN should be handled correctly."""
        np.random.seed(42)
        data = np.random.rand(20, 20) * 100
        data[:, 10] = np.nan

        result = apply_gaussian_filter(data, (5.0, 5.0))

        # NaN column should remain NaN
        assert np.all(np.isnan(result[:, 10]))
        # Other columns should have values
        assert not np.all(np.isnan(result[:, 5]))

    def test_nan_border(self):
        """NaN border should be preserved."""
        data = np.full((20, 20), np.nan)
        data[5:15, 5:15] = np.random.rand(10, 10) * 100

        result = apply_gaussian_filter(data, (3.0, 3.0))

        # NaN border should remain NaN
        assert np.all(np.isnan(result[:5, :]))
        assert np.all(np.isnan(result[15:, :]))
        assert np.all(np.isnan(result[:, :5]))
        assert np.all(np.isnan(result[:, 15:]))

    def test_very_small_cutoff(self):
        """Very small cutoff should approximate identity."""
        np.random.seed(42)
        data = np.random.rand(20, 20) * 100

        result = apply_gaussian_filter(data, (0.1, 0.1))

        # With very small cutoff, result should be close to original
        # (at least in the interior, away from edges)
        np.testing.assert_allclose(result[5:15, 5:15], data[5:15, 5:15], rtol=0.01)

    def test_very_large_cutoff(self):
        """Very large cutoff should approximate mean."""
        np.random.seed(42)
        data = np.random.rand(20, 20) * 100

        result = apply_gaussian_filter(data, (100.0, 100.0))

        # With very large cutoff, interior values should approach global mean
        # (edges are affected by zero-padding so we check interior)
        interior = result[8:12, 8:12]

        # Interior mean should be close to data mean (within reason, due to edge effects)
        assert np.mean(interior) == pytest.approx(np.mean(data), rel=0.5)

        # All interior values should be close to each other
        assert np.std(interior) < 1.0

    def test_negative_values(self):
        """Should handle negative values correctly."""
        np.random.seed(42)
        data = np.random.rand(20, 20) * 200 - 100  # Range [-100, 100]

        matlab_result = matlab_filter(data, 5.0)
        our_result = apply_gaussian_filter(data, (5.0, 5.0))

        np.testing.assert_allclose(our_result, matlab_result, rtol=1e-10)


class TestNumericalProperties:
    """Test numerical properties of the filter."""

    def test_kernel_sums_to_one(self):
        """Gaussian kernel should sum to 1 (normalization)."""
        for cutoff in [3, 5, 7, 10]:
            kernel = matlab_kernel_gauss(cutoff)
            assert np.sum(kernel) == pytest.approx(1.0, rel=1e-10)

    def test_kernel_is_symmetric(self):
        """Gaussian kernel should be symmetric."""
        kernel = matlab_kernel_gauss(5.0)

        # Symmetric about center
        np.testing.assert_allclose(kernel, kernel[::-1, :], rtol=1e-15)
        np.testing.assert_allclose(kernel, kernel[:, ::-1], rtol=1e-15)
        np.testing.assert_allclose(kernel, kernel.T, rtol=1e-15)

    def test_kernel_positive(self):
        """Gaussian kernel values should all be positive."""
        for cutoff in [3, 5, 7, 10]:
            kernel = matlab_kernel_gauss(cutoff)
            assert np.all(kernel > 0)

    def test_kernel_center_is_maximum(self):
        """Gaussian kernel center should be the maximum value."""
        kernel = matlab_kernel_gauss(5.0)
        center = kernel.shape[0] // 2
        assert kernel[center, center] == np.max(kernel)

    def test_output_in_input_range(self):
        """Filter output should be within input range (for non-edge pixels)."""
        np.random.seed(42)
        data = np.random.rand(50, 50) * 100
        result = apply_gaussian_filter(data, (5.0, 5.0))

        # Interior values should be within input range
        interior = result[10:40, 10:40]
        assert np.min(interior) >= np.min(data) - 1e-10
        assert np.max(interior) <= np.max(data) + 1e-10
