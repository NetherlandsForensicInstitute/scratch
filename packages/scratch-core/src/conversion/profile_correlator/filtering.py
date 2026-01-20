"""1D Gaussian filtering for profile data.

This module provides functions for applying Gaussian filters to 1D profiles
with proper handling of NaN values. The filtering uses normalized convolution
to ensure NaN values do not corrupt the filtered result.

The main functions are:
- cutoff_to_gaussian_sigma: Convert cutoff wavelength to Gaussian sigma
- apply_lowpass_filter_1d: Low-pass Gaussian filter
- apply_highpass_filter_1d: High-pass filter (removes shape)
- convolve_with_nan_handling: NaN-safe convolution

These correspond to the MATLAB functions:
- ChebyCutoffToGaussSigma.m
- ApplyLowPassFilter.m
- RemoveNoiseGaussian.m (uses apply_lowpass_filter_1d)
- RemoveShapeGaussian.m (uses apply_highpass_filter_1d)
- NanConv.m
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
# Which equals: cutoff_um / pixel_um * sqrt(2*ln(2))/(2*pi)
CHEBY_TO_GAUSS_FACTOR: float = 0.187390625


def cutoff_to_gaussian_sigma(
    cutoff_wavelength_um: float,
    pixel_size_m: float,
) -> float:
    """
    Convert cutoff wavelength to Gaussian sigma in samples.

    This function converts a cutoff wavelength (in micrometers) to the
    equivalent Gaussian filter sigma (in number of samples). The conversion
    is based on matching the 50% amplitude response of a Chebyshev filter.

    The formula used is::

        sigma = cutoff_um / pixel_um * sqrt(2*ln(2))/(2*pi)
              = cutoff_um / pixel_um * 0.187390625

    This corresponds to MATLAB's ChebyCutoffToGaussSigma.m.

    :param cutoff_wavelength_um: Cutoff wavelength in micrometers.
    :param pixel_size_m: Distance between samples in meters.
    :returns: Gaussian sigma in number of samples.

    Example::

        >>> # 100 um cutoff with 0.5 um pixel size
        >>> sigma = cutoff_to_gaussian_sigma(100.0, 0.5e-6)
        >>> sigma  # Should be about 37.5 samples
    """
    # Convert pixel size from meters to micrometers
    pixel_size_um = pixel_size_m * 1e6

    # Apply the conversion factor
    # MATLAB: sigma = cutoff/xdim * 0.187390625
    sigma = cutoff_wavelength_um / pixel_size_um * CHEBY_TO_GAUSS_FACTOR

    return sigma


def _create_gaussian_kernel_1d(
    sigma: float, n_sigma: float = 3.0
) -> NDArray[np.floating]:
    """
    Create a 1D Gaussian kernel.

    The kernel extends from -n_sigma*sigma to +n_sigma*sigma and is
    normalized to sum to 1.0.

    This replicates the kernel creation in MATLAB's ApplyLowPassFilter.m:
    - L = 1 + 2*round(alpha*sigma)  where alpha=3
    - n = (0:L)' - L/2
    - t = exp(-(1/2)*(alpha*n/(L/2))^2)
    - t = t / sum(t)

    :param sigma: Standard deviation in samples.
    :param n_sigma: Number of standard deviations to include (default 3.0).
    :returns: Normalized 1D Gaussian kernel.
    """
    # Calculate kernel length (must be odd for symmetry)
    # MATLAB: L = 1+2*round(alpha*sigma)
    half_length = int(round(n_sigma * sigma))
    kernel_length = 1 + 2 * half_length

    # Create coordinate vector centered at 0
    # MATLAB: L = L-1; n = (0:L)'-L/2
    n = np.arange(kernel_length) - half_length

    # Create Gaussian kernel
    # MATLAB: t = exp(-(1/2)*(alpha*n/(L/2)).^2)
    # The formula normalizes n by L/2 = half_length, then multiplies by alpha
    # This gives: t = exp(-0.5 * (3 * n / half_length)^2)
    #
    # Standard Gaussian would be: exp(-0.5 * (n/sigma)^2)
    # With the MATLAB formula: effective_sigma = half_length / n_sigma = sigma (approximately)
    if half_length > 0:
        normalized_coords = n_sigma * n / half_length
        kernel = np.exp(-0.5 * normalized_coords**2)
    else:
        # Edge case: very small sigma, single-sample kernel
        kernel = np.array([1.0])

    # Normalize to sum to 1
    # MATLAB: t = t./sum(t)
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

    This corresponds to MATLAB's NanConv.m with 'nanout' and 'edge' options.

    :param data: 1D input array. May contain NaN values.
    :param kernel: Convolution kernel. Must not contain NaN values.
    :param preserve_nan: If True, output has NaN where input had NaN.
    :param edge_correction: If True, apply edge correction to compensate
        for boundary effects from zero-padding.
    :returns: Convolved array with same shape as data.
    :raises ValueError: If kernel contains NaN values.

    Example::

        >>> data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        >>> kernel = np.array([0.25, 0.5, 0.25])
        >>> result = convolve_with_nan_handling(data, kernel)
        >>> np.isnan(result[1])  # NaN preserved at index 1
        True
    """
    data = np.asarray(data).ravel()
    kernel = np.asarray(kernel).ravel()

    # Validate kernel
    if np.any(np.isnan(kernel)):
        raise ValueError("Kernel must not contain NaN values.")

    # Identify NaN positions
    nan_mask = np.isnan(data)

    # Replace NaNs with zeros for convolution
    # MATLAB: a(n) = 0; on(n) = 0;
    data_filled = np.where(nan_mask, 0.0, data)
    weights = np.where(nan_mask, 0.0, 1.0)

    # Convolve data and weights
    # MATLAB uses conv2 with 'same' mode
    # Cast to float64 arrays to ensure proper typing
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
        # MATLAB: if(any(n(:)) && ~edge); flat = flat./conv2(o,k,shape); end
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
    # MATLAB: if(nanout); c(n) = NaN; end
    if preserve_nan:
        result[nan_mask] = np.nan

    return result


def apply_lowpass_filter_1d(
    profile: NDArray[np.floating],
    cutoff_wavelength_um: float,
    pixel_size_m: float,
    cut_borders: bool = False,
) -> NDArray[np.floating]:
    """
    Apply Gaussian low-pass filter to a 1D profile with NaN handling.

    This function smooths a profile by removing high-frequency components
    (noise) while preserving low-frequency components (shape). It handles
    NaN values using normalized convolution.

    The filter is created with a Gaussian kernel extending from -3*sigma
    to +3*sigma, where sigma is computed from the cutoff wavelength.

    This corresponds to MATLAB's ApplyLowPassFilter.m and RemoveNoiseGaussian.m.

    :param profile: 1D array of heights. May contain NaN values.
    :param cutoff_wavelength_um: Filter cutoff wavelength in micrometers.
        Frequency components with wavelengths shorter than this are attenuated.
    :param pixel_size_m: Sample spacing in meters.
    :param cut_borders: If True, trim filter-affected borders by sigma samples
        from each end.
    :returns: Low-pass filtered profile. Same length as input unless
        cut_borders=True.

    Example::

        >>> import numpy as np
        >>> # Create noisy profile
        >>> x = np.linspace(0, 10, 1000)
        >>> profile = np.sin(x) + 0.1 * np.random.randn(1000)
        >>> # Apply 50 um low-pass filter with 0.5 um pixels
        >>> filtered = apply_lowpass_filter_1d(profile, 50.0, 0.5e-6)
    """
    profile = np.asarray(profile).ravel()

    # Convert cutoff to Gaussian sigma
    sigma = cutoff_to_gaussian_sigma(cutoff_wavelength_um, pixel_size_m)

    # Create Gaussian kernel
    # MATLAB: alpha = 3; L = 1+2*round(alpha*sigma)
    kernel = _create_gaussian_kernel_1d(sigma, n_sigma=3.0)

    # Apply convolution with NaN handling
    # MATLAB: profile_out = NanConv(profile, t, 'nanout', 'edge')
    filtered = convolve_with_nan_handling(
        profile, kernel, preserve_nan=True, edge_correction=True
    )

    # Optionally cut borders affected by filter
    if cut_borders:
        # MATLAB: sigma = round(sigma); profile_out = profile_out(1+sigma:end-sigma)
        border = int(round(sigma))
        if border > 0 and len(filtered) > 2 * border:
            filtered = filtered[border:-border]

    return filtered


def apply_highpass_filter_1d(
    profile: NDArray[np.floating],
    cutoff_wavelength_um: float,
    pixel_size_m: float,
    cut_borders: bool = False,
) -> NDArray[np.floating]:
    """
    Apply Gaussian high-pass filter to a 1D profile.

    This function removes low-frequency components (shape) while preserving
    high-frequency components (detail/noise). It is computed as::

        highpass = original - lowpass(original, cutoff)

    This corresponds to MATLAB's RemoveShapeGaussian.m.

    :param profile: 1D array of heights. May contain NaN values.
    :param cutoff_wavelength_um: Filter cutoff wavelength in micrometers.
        Frequency components with wavelengths longer than this are removed.
    :param pixel_size_m: Sample spacing in meters.
    :param cut_borders: If True, trim filter-affected borders by sigma samples
        from each end.
    :returns: High-pass filtered profile. Same length as input unless
        cut_borders=True.

    Example::

        >>> import numpy as np
        >>> # Create profile with shape and detail
        >>> x = np.linspace(0, 10, 1000)
        >>> profile = x**2 + np.sin(10*x)  # Quadratic shape + high-freq detail
        >>> # Remove shape (wavelengths > 100 um)
        >>> detail = apply_highpass_filter_1d(profile, 100.0, 0.5e-6)
    """
    profile = np.asarray(profile).ravel()

    # Compute low-pass filtered version
    lowpass = apply_lowpass_filter_1d(
        profile, cutoff_wavelength_um, pixel_size_m, cut_borders=False
    )

    # High-pass = original - low-pass
    highpass = profile - lowpass

    # Optionally cut borders affected by filter
    if cut_borders:
        sigma = cutoff_to_gaussian_sigma(cutoff_wavelength_um, pixel_size_m)
        border = int(round(sigma))
        if border > 0 and len(highpass) > 2 * border:
            highpass = highpass[border:-border]

    return highpass
