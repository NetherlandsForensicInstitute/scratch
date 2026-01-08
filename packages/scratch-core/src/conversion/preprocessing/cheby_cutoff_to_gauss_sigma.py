"""
Convert Chebyshev filter cutoff wavelength to Gaussian filter sigma.

This module provides the conversion from Chebyshev filter cutoff wavelength
to Gaussian filter sigma parameter, following the ISO standard for Gaussian
filters used in surface metrology.
"""

import math

# This ensures the Gaussian filter satisfies the ISO 16610-21 standard
# for surface texture analysis.
ISO_GAUSSIAN_CONSTANT = math.sqrt(2 * math.log(2)) / (2 * math.pi)


def cheby_cutoff_to_gauss_sigma(cutoff_um: float, xdim_m: float) -> float:
    """
    Convert Chebyshev cutoff wavelength to Gaussian filter sigma.

    This function converts a Chebyshev filter cutoff wavelength (commonly used
    in surface metrology standards) to the equivalent sigma parameter for a
    Gaussian filter. The conversion uses the ISO standard constant that ensures
    the Gaussian filter meets the ISO 16610-21 specification.

    The mathematical relationship is:
        sigma = (cutoff / xdim) * sqrt(2 * ln(2)) / (2 * pi)

    Where:
        - cutoff is the cutoff wavelength
        - xdim is the sampling interval (pixel spacing)
        - The constant sqrt(2 * ln(2)) / (2 * pi) ≈ 0.187390625

    :param cutoff_um: Cutoff wavelength in micrometers (um). This defines the
        spatial frequency at which the filter attenuates the signal by 50%.
        Common values are 250 um (noise removal) or 2000 um (shape removal).
    :param xdim_m: Distance between measurements (pixel spacing) in meters (m).
        This is the physical distance between adjacent pixels in the scan.
    :return: Sigma of the Gaussian filter in pixels. This value can be used
        directly with scipy.ndimage.gaussian_filter or similar functions.
    :raises ValueError: If xdim_m is zero or negative.

    Example:
        >>> # Typical scanner with 0.438 um pixel spacing, 2000 um cutoff
        >>> sigma = cheby_cutoff_to_gauss_sigma(2000.0, 438e-9)
        >>> print(f"Sigma: {sigma:.2f} pixels")
        Sigma: 856.12 pixels

    Note:
        The cutoff wavelength determines what spatial frequencies are filtered:
        - Shape removal (highpass): Use large cutoff (e.g., 2000 um) to remove
          low-frequency form/curvature while preserving fine detail.
        - Noise removal (lowpass): Use small cutoff (e.g., 250 um) to remove
          high-frequency noise while preserving the signal of interest.
    """
    if xdim_m <= 0:
        raise ValueError(f"xdim_m must be positive, got {xdim_m}")

    # Convert xdim from meters to micrometers to match cutoff units
    xdim_um = xdim_m * 1e6

    # Calculate sigma using the ISO standard constant
    # sigma = (cutoff / xdim) * constant
    # Result is in pixels (dimensionless ratio scaled by constant)
    sigma = (cutoff_um / xdim_um) * ISO_GAUSSIAN_CONSTANT

    return sigma


def gauss_sigma_to_cheby_cutoff(sigma: float, xdim_m: float) -> float:
    """
    Convert Gaussian filter sigma back to Chebyshev cutoff wavelength.

    This is the inverse of cheby_cutoff_to_gauss_sigma, useful for determining
    what cutoff wavelength corresponds to a given sigma value.

    :param sigma: Sigma of the Gaussian filter in pixels.
    :param xdim_m: Distance between measurements (pixel spacing) in meters (m).
    :return: Cutoff wavelength in micrometers (um).
    :raises ValueError: If xdim_m is zero or negative.

    Example:
        >>> # Given sigma = 100 pixels and 0.438 um pixel spacing
        >>> cutoff = gauss_sigma_to_cheby_cutoff(100.0, 438e-9)
        >>> print(f"Cutoff: {cutoff:.1f} um")
        Cutoff: 233.7 um
    """
    if xdim_m <= 0:
        raise ValueError(f"xdim_m must be positive, got {xdim_m}")

    # Convert xdim from meters to micrometers
    xdim_um = xdim_m * 1e6

    # Inverse of the forward calculation
    cutoff_um = (sigma / ISO_GAUSSIAN_CONSTANT) * xdim_um

    return cutoff_um
