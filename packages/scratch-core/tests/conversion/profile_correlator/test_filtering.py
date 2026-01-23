"""Tests for the filtering module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from conversion.profile_correlator import (
    CHEBY_TO_GAUSS_FACTOR,
    cutoff_to_gaussian_sigma,
    apply_lowpass_filter_1d,
    apply_highpass_filter_1d,
    convolve_with_nan_handling,
)


class TestCutoffToGaussianSigma:
    """Tests for cutoff_to_gaussian_sigma function."""

    def test_known_conversion(self):
        """Test conversion with known values."""
        # 100 um cutoff with 0.5 um pixel size (all in meters)
        # sigma = 100e-6 / 0.5e-6 * 0.187390625 = 200 * 0.187390625 = 37.478125
        result = cutoff_to_gaussian_sigma(100e-6, 0.5e-6)
        assert_allclose(result, 37.478125, atol=1e-6)

    def test_proportional_to_cutoff(self):
        """Sigma should be proportional to cutoff wavelength."""
        sigma_50 = cutoff_to_gaussian_sigma(50e-6, 0.5e-6)
        sigma_100 = cutoff_to_gaussian_sigma(100e-6, 0.5e-6)
        assert_allclose(sigma_100 / sigma_50, 2.0, atol=1e-10)

    def test_inversely_proportional_to_pixel_size(self):
        """Sigma should be inversely proportional to pixel size."""
        sigma_05 = cutoff_to_gaussian_sigma(100e-6, 0.5e-6)
        sigma_1 = cutoff_to_gaussian_sigma(100e-6, 1.0e-6)
        assert_allclose(sigma_05 / sigma_1, 2.0, atol=1e-10)

    def test_constant_matches_matlab(self):
        """The conversion constant should match MATLAB's value."""
        # sqrt(2*ln(2))/(2*pi) ≈ 0.187390625
        expected = np.sqrt(2 * np.log(2)) / (2 * np.pi)
        assert_allclose(CHEBY_TO_GAUSS_FACTOR, expected, atol=1e-6)


class TestConvolveWithNanHandling:
    """Tests for convolve_with_nan_handling function."""

    def test_simple_convolution_without_nans(self):
        """Basic convolution should work without NaN values."""
        data = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        kernel = np.array([0.25, 0.5, 0.25])
        result = convolve_with_nan_handling(data, kernel)

        # Result should be smoothed version of delta function
        assert len(result) == len(data)
        assert result[2] > result[1]  # Peak at center
        assert result[2] > result[3]

    def test_nan_positions_preserved(self):
        """NaN positions should be preserved in output."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        kernel = np.array([0.25, 0.5, 0.25])
        result = convolve_with_nan_handling(data, kernel, preserve_nan=True)

        assert np.isnan(result[2])
        assert not np.isnan(result[0])
        assert not np.isnan(result[4])

    def test_nan_not_preserved_option(self):
        """When preserve_nan=False, NaN positions should be filled."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        kernel = np.array([0.25, 0.5, 0.25])
        result = convolve_with_nan_handling(data, kernel, preserve_nan=False)

        # Should interpolate across the NaN
        assert not np.isnan(result[2])

    def test_normalized_convolution(self):
        """With NaN values, result should use normalized convolution."""
        data = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
        kernel = np.array([1.0, 1.0, 1.0])
        result = convolve_with_nan_handling(data, kernel, preserve_nan=False)

        # For constant data, result should be constant (= 1.0)
        # NaN positions get interpolated
        for i, val in enumerate(result):
            if not np.isnan(data[i]):
                assert_allclose(val, 1.0, atol=1e-10)

    def test_kernel_with_nan_raises_error(self):
        """Kernel containing NaN should raise ValueError."""
        data = np.array([1.0, 2.0, 3.0])
        kernel = np.array([0.25, np.nan, 0.25])
        with pytest.raises(ValueError):
            convolve_with_nan_handling(data, kernel)

    def test_edge_correction(self):
        """Edge correction should compensate for boundary effects."""
        data = np.ones(10)
        kernel = np.array([0.25, 0.5, 0.25])
        result = convolve_with_nan_handling(data, kernel, edge_correction=True)

        # With edge correction, constant signal should remain constant
        assert_allclose(result, 1.0, atol=1e-10)


class TestApplyLowpassFilter1d:
    """Tests for apply_lowpass_filter_1d function."""

    def test_removes_high_frequency_noise(self):
        """Low-pass filter should attenuate high-frequency components."""
        np.random.seed(42)
        n = 1000
        pixel_size = 0.5e-6  # 0.5 um

        # Create signal with low and high frequency components
        x = np.arange(n) * pixel_size * 1e6  # x in um
        low_freq = np.sin(2 * np.pi * x / 100)  # 100 um wavelength
        high_freq = 0.5 * np.sin(2 * np.pi * x / 5)  # 5 um wavelength (noise)
        signal = low_freq + high_freq

        # Filter with 20 um cutoff (should remove 5 um component)
        filtered = apply_lowpass_filter_1d(signal, 20e-6, pixel_size)

        # High frequency amplitude should be reduced
        # Compare variance of difference from low_freq
        original_diff_var = np.var(signal - low_freq)
        filtered_diff_var = np.var(filtered - low_freq)
        assert filtered_diff_var < original_diff_var * 0.5

    def test_preserves_nan_values(self):
        """NaN values should be preserved in the output."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 20, dtype=float)
        result = apply_lowpass_filter_1d(data, 50e-6, 0.5e-6)

        # Check NaN positions are preserved
        nan_positions = np.isnan(data)
        result_nan_positions = np.isnan(result)
        np.testing.assert_array_equal(nan_positions, result_nan_positions)

    def test_output_length_matches_input(self):
        """Output should have same length as input."""
        data = np.random.randn(500)
        result = apply_lowpass_filter_1d(data, 50e-6, 0.5e-6)
        assert len(result) == len(data)

    def test_cut_borders_option(self):
        """cut_borders=True should trim the output."""
        data = np.random.randn(500)
        result = apply_lowpass_filter_1d(data, 50e-6, 0.5e-6, cut_borders=True)
        assert len(result) < len(data)


class TestApplyHighpassFilter1d:
    """Tests for apply_highpass_filter_1d function."""

    def test_removes_low_frequency_shape(self):
        """High-pass filter should remove low-frequency components."""
        n = 1000
        pixel_size = 0.5e-6  # 0.5 um

        # Create signal with low and high frequency components
        x = np.arange(n) * pixel_size * 1e6  # x in um
        low_freq = np.sin(2 * np.pi * x / 500)  # 500 um wavelength (shape)
        high_freq = 0.3 * np.sin(2 * np.pi * x / 20)  # 20 um wavelength (detail)
        signal = low_freq + high_freq

        # Filter with 100 um cutoff (should remove 500 um component)
        filtered = apply_highpass_filter_1d(signal, 100e-6, pixel_size)

        # Low frequency amplitude should be reduced
        # The filtered signal should have similar variance to high_freq alone
        assert np.var(filtered) < np.var(signal) * 0.5

    def test_highpass_is_original_minus_lowpass(self):
        """High-pass should equal original minus low-pass filtered."""
        np.random.seed(42)
        data = np.random.randn(200)
        cutoff = 50e-6
        pixel_size = 0.5e-6

        lowpass = apply_lowpass_filter_1d(data, cutoff, pixel_size)
        highpass = apply_highpass_filter_1d(data, cutoff, pixel_size)

        expected = data - lowpass
        assert_allclose(highpass, expected, atol=1e-10)

    def test_preserves_nan_values(self):
        """NaN values should be preserved in the output."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 20, dtype=float)
        result = apply_highpass_filter_1d(data, 50e-6, 0.5e-6)

        nan_positions = np.isnan(data)
        result_nan_positions = np.isnan(result)
        np.testing.assert_array_equal(nan_positions, result_nan_positions)
