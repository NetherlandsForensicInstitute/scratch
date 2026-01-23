"""
1D Gaussian filtering for profile data.

This module provides functions for applying Gaussian filters to 1D profiles
with proper handling of NaN values. The filtering uses normalized convolution
to ensure NaN values do not corrupt the filtered result.

The main functions are:
- cutoff_to_gaussian_sigma: Convert cutoff wavelength to Gaussian sigma
- apply_lowpass_filter_1d: Low-pass Gaussian filter
- apply_highpass_filter_1d: High-pass filter (removes shape)
- convolve_with_nan_handling: NaN-safe convolution

All length parameters are in meters (SI units).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve

# Constant for converting Chebyshev cutoff to Gaussian sigma
# This is sqrt(2*ln(2))/(2*pi) ≈ 0.187390625
# Used in MATLAB's ChebyCutoffToGaussSigma.m
#
# Derivation: For a Gaussian filter to have 50% amplitude response at
# the cutoff frequency f_c:
#   H(f_c) = exp(-2 * (pi * sigma * f_c)^2) = 0.5
#   sigma = sqrt(ln(2)) / (pi * f_c) = sqrt(ln(2)) / pi * lambda_c
# where lambda_c = 1/f_c is the cutoff wavelength.
#
# In samples: sigma_samples = lambda_c_samples * sqrt(ln(2)) / pi
# Which equals: cutoff / pixel_size * sqrt(2*ln(2))/(2*pi)
CHEBY_TO_GAUSS_FACTOR: float = 0.187390625


def cutoff_to_gaussian_sigma(
    cutoff_wavelength: float,
    pixel_size: float,
) -> float:
    """
    Convert cutoff wavelength to Gaussian sigma in samples.

    This function converts a cutoff wavelength to the equivalent Gaussian
    filter sigma (in number of samples). The conversion is based on matching
    the 50% amplitude response of a Chebyshev filter.

    The formula used is::

        sigma = cutoff / pixel_size * sqrt(2*ln(2))/(2*pi)
              = cutoff / pixel_size * 0.187390625

    :param cutoff_wavelength: Cutoff wavelength in meters.
    :param pixel_size: Distance between samples in meters.
    :returns: Gaussian sigma in number of samples.
    """
    # Apply the conversion factor (both in same units, so ratio is dimensionless)
    sigma = cutoff_wavelength / pixel_size * CHEBY_TO_GAUSS_FACTOR

    return sigma


def _create_gaussian_kernel_1d(
    sigma: float, n_sigma: float = 3.0
) -> NDArray[np.floating]:
    """
    Create a 1D Gaussian kernel.

    The kernel extends from -n_sigma*sigma to +n_sigma*sigma and is
    normalized to sum to 1.0.

    :param sigma: Standard deviation in samples.
    :param n_sigma: Number of standard deviations to include (default 3.0).
    :returns: Normalized 1D Gaussian kernel.
    """
    # Calculate kernel length (must be odd for symmetry)
    half_length = int(round(n_sigma * sigma))
    kernel_length = 1 + 2 * half_length

    # Create coordinate vector centered at 0
    n = np.arange(kernel_length) - half_length

    # Create Gaussian kernel
    # The formula normalizes n by L/2 = half_length, then multiplies by alpha
    # This gives: t = exp(-0.5 * (3 * n / half_length)^2)
    #
    # Standard Gaussian would be: exp(-0.5 * (n/sigma)^2)
    if half_length > 0:
        normalized_coords = n_sigma * n / half_length
        kernel = np.exp(-0.5 * normalized_coords**2)
    else:
        # Edge case: very small sigma, single-sample kernel
        kernel = np.array([1.0])

    # Normalize to sum to 1
    kernel = kernel / np.sum(kernel)

    return kernel


def convolve_with_nan_handling(
    data: NDArray[np.floating],
    kernel: NDArray[np.floating],
    preserve_nan: bool = True,
    edge_correction: bool = True,
) -> NDArray[np.floating]:
    """
    Convolve data with kernel, properly handling NaN values.

    This function performs normalized convolution where NaN values are treated
    as missing data with zero weight. The result is computed as::

        result = convolve(data_filled, kernel) / convolve(weights, kernel)

    where data_filled has NaNs replaced with zeros, and weights are 1 where
    data is valid and 0 where data is NaN.

    Edge correction ensures that edge effects from zero-padding are properly
    accounted for (the result is normalized by the actual weight contribution).

    :param data: 1D input array. May contain NaN values.
    :param kernel: Convolution kernel. Must not contain NaN values.
    :param preserve_nan: If True, output has NaN where input had NaN.
    :param edge_correction: If True, apply edge correction to compensate
        for boundary effects from zero-padding.
    :returns: Convolved array with same shape as data.
    :raises ValueError: If kernel contains NaN values.
    """
    data = np.asarray(data).ravel()
    kernel = np.asarray(kernel).ravel()

    # Validate kernel
    if np.any(np.isnan(kernel)):
        raise ValueError("Kernel must not contain NaN values.")

    # Identify NaN positions
    nan_mask = np.isnan(data)

    # Replace NaNs with zeros for convolution
    data_filled = np.where(nan_mask, 0.0, data)
    weights = np.where(nan_mask, 0.0, 1.0)

    # Convolve data and weights
    numerator: NDArray[np.floating] = np.asarray(
        convolve(data_filled, kernel, mode="same"), dtype=np.float64
    )
    denominator: NDArray[np.floating] = np.asarray(
        convolve(weights, kernel, mode="same"), dtype=np.float64
    )

    # Apply edge correction if requested
    # Without edge correction, we'd divide by convolve(ones, kernel)
    # With edge correction (edge=True), we use the actual weight sum
    if edge_correction:
        # The denominator already accounts for both NaN positions and edges
        pass
    else:
        # For no edge correction, also normalize by what a full kernel would give
        full_weights: NDArray[np.floating] = np.asarray(
            convolve(np.ones_like(data), kernel, mode="same"), dtype=np.float64
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = denominator / full_weights
            denominator = np.where(full_weights == 0, 0.0, denominator)

    # Compute normalized result
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        # Where denominator is 0, result is undefined (set to 0 or NaN)
        result = np.where(denominator == 0, np.nan, result)

    # Restore NaN positions if requested
    if preserve_nan:
        result[nan_mask] = np.nan

    return result


def apply_lowpass_filter_1d(
    profile: NDArray[np.floating],
    cutoff_wavelength: float,
    pixel_size: float,
    cut_borders: bool = False,
) -> NDArray[np.floating]:
    """
    Apply Gaussian low-pass filter to a 1D profile with NaN handling.

    This function smooths a profile by removing high-frequency components
    (noise) while preserving low-frequency components (shape). It handles
    NaN values using normalized convolution.

    The filter is created with a Gaussian kernel extending from -3*sigma
    to +3*sigma, where sigma is computed from the cutoff wavelength.

    :param profile: 1D array of heights. May contain NaN values.
    :param cutoff_wavelength: Filter cutoff wavelength in meters.
        Frequency components with wavelengths shorter than this are attenuated.
    :param pixel_size: Sample spacing in meters.
    :param cut_borders: If True, trim filter-affected borders by sigma samples
        from each end.
    :returns: Low-pass filtered profile. Same length as input unless
        cut_borders=True.
    """
    profile = np.asarray(profile).ravel()

    # Convert cutoff to Gaussian sigma
    sigma = cutoff_to_gaussian_sigma(cutoff_wavelength, pixel_size)

    # Create Gaussian kernel
    kernel = _create_gaussian_kernel_1d(sigma, n_sigma=3.0)

    # Apply convolution with NaN handling
    filtered = convolve_with_nan_handling(
        profile, kernel, preserve_nan=True, edge_correction=True
    )

    # Optionally cut borders affected by filter
    if cut_borders:
        border = int(round(sigma))
        if border > 0 and len(filtered) > 2 * border:
            filtered = filtered[border:-border]

    return filtered


def apply_highpass_filter_1d(
    profile: NDArray[np.floating],
    cutoff_wavelength: float,
    pixel_size: float,
    cut_borders: bool = False,
) -> NDArray[np.floating]:
    """
    Apply Gaussian high-pass filter to a 1D profile.

    This function removes low-frequency components (shape) while preserving
    high-frequency components (detail/noise). It is computed as::

        highpass = original - lowpass(original, cutoff)

    :param profile: 1D array of heights. May contain NaN values.
    :param cutoff_wavelength: Filter cutoff wavelength in meters.
        Frequency components with wavelengths longer than this are removed.
    :param pixel_size: Sample spacing in meters.
    :param cut_borders: If True, trim filter-affected borders by sigma samples
        from each end.
    :returns: High-pass filtered profile. Same length as input unless
        cut_borders=True.
    """
    profile = np.asarray(profile).ravel()

    # Compute low-pass filtered version
    lowpass = apply_lowpass_filter_1d(
        profile, cutoff_wavelength, pixel_size, cut_borders=False
    )

    # High-pass = original - low-pass
    highpass = profile - lowpass

    # Optionally cut borders affected by filter
    if cut_borders:
        sigma = cutoff_to_gaussian_sigma(cutoff_wavelength, pixel_size)
        border = int(round(sigma))
        if border > 0 and len(highpass) > 2 * border:
            highpass = highpass[border:-border]

    return highpass
