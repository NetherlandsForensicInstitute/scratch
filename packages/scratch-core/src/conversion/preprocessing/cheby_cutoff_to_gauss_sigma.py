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


def cheby_cutoff_to_gauss_sigma(cutoff: float, xdim: float) -> float:
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

    Parameters
    ----------
    cutoff : float
        Cutoff wavelength in meters (m). This defines the spatial frequency
        at which the filter attenuates the signal by 50%. Common values are
        250e-6 m (noise removal) or 2000e-6 m (shape removal).
    xdim : float
        Distance between measurements (pixel spacing) in meters (m).
        This is the physical distance between adjacent pixels in the scan.

    Returns
    -------
    float
        Sigma of the Gaussian filter in pixels. This value can be used
        directly with scipy.ndimage.gaussian_filter or similar functions.

    Raises
    ------
    ValueError
        If xdim is zero or negative.

    Example
    -------
    >>> # Typical scanner with 0.438 um pixel spacing, 2000 um cutoff
    >>> sigma = cheby_cutoff_to_gauss_sigma(2000e-6, 0.438e-6)
    >>> print(f"Sigma: {sigma:.2f} pixels")
    Sigma: 856.12 pixels

    Notes
    -----
    The cutoff wavelength determines what spatial frequencies are filtered:
    - Shape removal (highpass): Use large cutoff (e.g., 2000e-6 m) to remove
      low-frequency form/curvature while preserving fine detail.
    - Noise removal (lowpass): Use small cutoff (e.g., 250e-6 m) to remove
      high-frequency noise while preserving the signal of interest.
    """
    if xdim <= 0:
        raise ValueError(f"xdim must be positive, got {xdim}")

    # Calculate sigma using the ISO standard constant
    # sigma = (cutoff / xdim) * constant
    # Result is in pixels (dimensionless ratio scaled by constant)
    sigma = (cutoff / xdim) * ISO_GAUSSIAN_CONSTANT

    return sigma


def gauss_sigma_to_cheby_cutoff(sigma: float, xdim: float) -> float:
    """
    Convert Gaussian filter sigma back to Chebyshev cutoff wavelength.

    This is the inverse of cheby_cutoff_to_gauss_sigma, useful for determining
    what cutoff wavelength corresponds to a given sigma value.

    Parameters
    ----------
    sigma : float
        Sigma of the Gaussian filter in pixels.
    xdim : float
        Distance between measurements (pixel spacing) in meters (m).

    Returns
    -------
    float
        Cutoff wavelength in meters (m).

    Raises
    ------
    ValueError
        If xdim is zero or negative.

    Example
    -------
    >>> # Given sigma = 100 pixels and 0.438 um pixel spacing
    >>> cutoff = gauss_sigma_to_cheby_cutoff(100.0, 0.438e-6)
    >>> print(f"Cutoff: {cutoff*1e6:.1f} um")
    Cutoff: 233.7 um
    """
    if xdim <= 0:
        raise ValueError(f"xdim must be positive, got {xdim}")

    # Inverse of the forward calculation
    cutoff = (sigma / ISO_GAUSSIAN_CONSTANT) * xdim

    return cutoff
