"""Tests for Chebyshev cutoff to Gaussian sigma conversion.

This module tests the conversion between Chebyshev filter cutoff wavelength
and Gaussian filter sigma, verifying compliance with the ISO standard.
"""

import math

import pytest

from conversion.cheby_cutoff_to_gauss_sigma import (
    ISO_GAUSSIAN_CONSTANT,
    cheby_cutoff_to_gauss_sigma,
    gauss_sigma_to_cheby_cutoff,
)


class TestIsoGaussianConstant:
    """Test the ISO Gaussian constant value."""

    def test_constant_matches_mathematical_definition(self) -> None:
        """Verify the constant equals sqrt(2*ln(2)) / (2*pi)."""
        expected = math.sqrt(2 * math.log(2)) / (2 * math.pi)
        assert ISO_GAUSSIAN_CONSTANT == pytest.approx(expected, rel=1e-9)

    def test_constant_value(self) -> None:
        """Verify the hardcoded constant value."""
        assert ISO_GAUSSIAN_CONSTANT == pytest.approx(0.187390625, rel=1e-9)


class TestChebyCutoffToGaussSigma:
    """Test conversion from Chebyshev cutoff to Gaussian sigma."""

    def test_typical_shape_removal_parameters(self) -> None:
        """Test with typical shape removal parameters (2000 um cutoff)."""
        # Typical Alicona scanner: 0.438 um pixel spacing
        xdim_m = 438e-9  # 0.438 um in meters
        cutoff_um = 2000.0

        sigma = cheby_cutoff_to_gauss_sigma(cutoff_um, xdim_m)

        # Expected: (2000 / 0.438) * 0.187390625 ≈ 855.66
        expected = (cutoff_um / (xdim_m * 1e6)) * ISO_GAUSSIAN_CONSTANT
        assert sigma == pytest.approx(expected, rel=1e-9)
        assert sigma > 0

    def test_typical_noise_removal_parameters(self) -> None:
        """Test with typical noise removal parameters (250 um cutoff)."""
        xdim_m = 438e-9
        cutoff_um = 250.0

        sigma = cheby_cutoff_to_gauss_sigma(cutoff_um, xdim_m)

        expected = (cutoff_um / (xdim_m * 1e6)) * ISO_GAUSSIAN_CONSTANT
        assert sigma == pytest.approx(expected, rel=1e-9)

    def test_sigma_scales_linearly_with_cutoff(self) -> None:
        """Verify sigma scales linearly with cutoff wavelength."""
        xdim_m = 1e-6  # 1 um pixel spacing

        sigma_1000 = cheby_cutoff_to_gauss_sigma(1000.0, xdim_m)
        sigma_2000 = cheby_cutoff_to_gauss_sigma(2000.0, xdim_m)

        assert sigma_2000 == pytest.approx(sigma_1000 * 2, rel=1e-9)

    def test_sigma_scales_inversely_with_xdim(self) -> None:
        """Verify sigma scales inversely with pixel spacing."""
        cutoff_um = 1000.0

        sigma_1um = cheby_cutoff_to_gauss_sigma(cutoff_um, 1e-6)
        sigma_2um = cheby_cutoff_to_gauss_sigma(cutoff_um, 2e-6)

        assert sigma_1um == pytest.approx(sigma_2um * 2, rel=1e-9)

    def test_zero_cutoff_returns_zero(self) -> None:
        """Zero cutoff wavelength should return zero sigma."""
        sigma = cheby_cutoff_to_gauss_sigma(0.0, 1e-6)
        assert sigma == 0.0

    def test_very_small_cutoff(self) -> None:
        """Test with very small cutoff values."""
        sigma = cheby_cutoff_to_gauss_sigma(1.0, 1e-6)
        expected = 1.0 * ISO_GAUSSIAN_CONSTANT
        assert sigma == pytest.approx(expected, rel=1e-9)

    def test_very_large_cutoff(self) -> None:
        """Test with very large cutoff values."""
        sigma = cheby_cutoff_to_gauss_sigma(10000.0, 1e-6)
        expected = 10000.0 * ISO_GAUSSIAN_CONSTANT
        assert sigma == pytest.approx(expected, rel=1e-9)

    def test_zero_xdim_raises_error(self) -> None:
        """Zero pixel spacing should raise ValueError."""
        with pytest.raises(ValueError, match="xdim_m must be positive"):
            cheby_cutoff_to_gauss_sigma(1000.0, 0.0)

    def test_negative_xdim_raises_error(self) -> None:
        """Negative pixel spacing should raise ValueError."""
        with pytest.raises(ValueError, match="xdim_m must be positive"):
            cheby_cutoff_to_gauss_sigma(1000.0, -1e-6)

    def test_various_pixel_spacings(self) -> None:
        """Test with various common pixel spacings."""
        cutoff_um = 1000.0

        # Test different scanner resolutions
        pixel_spacings = [
            438e-9,  # Alicona typical
            1e-6,  # 1 um
            500e-9,  # 0.5 um
            2e-6,  # 2 um
        ]

        for xdim_m in pixel_spacings:
            sigma = cheby_cutoff_to_gauss_sigma(cutoff_um, xdim_m)
            assert sigma > 0
            # Verify formula
            xdim_um = xdim_m * 1e6
            expected = (cutoff_um / xdim_um) * ISO_GAUSSIAN_CONSTANT
            assert sigma == pytest.approx(expected, rel=1e-9)


class TestGaussSigmaToCutoff:
    """Test inverse conversion from Gaussian sigma to Chebyshev cutoff."""

    def test_roundtrip_conversion(self) -> None:
        """Converting cutoff -> sigma -> cutoff should return original value."""
        xdim_m = 438e-9
        original_cutoff = 2000.0

        sigma = cheby_cutoff_to_gauss_sigma(original_cutoff, xdim_m)
        recovered_cutoff = gauss_sigma_to_cheby_cutoff(sigma, xdim_m)

        assert recovered_cutoff == pytest.approx(original_cutoff, rel=1e-9)

    def test_inverse_roundtrip(self) -> None:
        """Converting sigma -> cutoff -> sigma should return original value."""
        xdim_m = 1e-6
        original_sigma = 100.0

        cutoff = gauss_sigma_to_cheby_cutoff(original_sigma, xdim_m)
        recovered_sigma = cheby_cutoff_to_gauss_sigma(cutoff, xdim_m)

        assert recovered_sigma == pytest.approx(original_sigma, rel=1e-9)

    def test_zero_xdim_raises_error(self) -> None:
        """Zero pixel spacing should raise ValueError."""
        with pytest.raises(ValueError, match="xdim_m must be positive"):
            gauss_sigma_to_cheby_cutoff(100.0, 0.0)

    def test_negative_xdim_raises_error(self) -> None:
        """Negative pixel spacing should raise ValueError."""
        with pytest.raises(ValueError, match="xdim_m must be positive"):
            gauss_sigma_to_cheby_cutoff(100.0, -1e-6)

    def test_various_sigma_values(self) -> None:
        """Test conversion with various sigma values."""
        xdim_m = 1e-6

        for sigma in [10.0, 50.0, 100.0, 500.0, 1000.0]:
            cutoff = gauss_sigma_to_cheby_cutoff(sigma, xdim_m)
            # Verify by roundtrip
            recovered = cheby_cutoff_to_gauss_sigma(cutoff, xdim_m)
            assert recovered == pytest.approx(sigma, rel=1e-9)


class TestMatlabCompatibility:
    """Test compatibility with MATLAB implementation."""

    def test_matlab_example_values(self) -> None:
        """Test with values from MATLAB ChebyCutoffToGaussSigma.m.

        MATLAB code:
            xdim  = xdim * 1e6; % [m] -> [um]
            sigma = cutoff/xdim * 0.187390625;
        """
        # Test case: cutoff = 2000 um, xdim = 438e-9 m
        cutoff_um = 2000.0
        xdim_m = 438e-9

        sigma = cheby_cutoff_to_gauss_sigma(cutoff_um, xdim_m)

        # Manual calculation matching MATLAB
        xdim_um = xdim_m * 1e6  # = 0.438
        expected = cutoff_um / xdim_um * 0.187390625  # = 855.66...

        assert sigma == pytest.approx(expected, rel=1e-9)

    def test_constant_matches_matlab_comment(self) -> None:
        """Verify constant matches MATLAB comment: sqrt(2ln(2))/2pi."""
        # From MATLAB: % 0.187.. = sqrt(2ln(2))/2pi
        matlab_constant = math.sqrt(2 * math.log(2)) / (2 * math.pi)
        assert ISO_GAUSSIAN_CONSTANT == pytest.approx(matlab_constant, rel=1e-9)
