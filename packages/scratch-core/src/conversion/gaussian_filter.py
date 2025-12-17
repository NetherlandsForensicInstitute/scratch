from functools import partial, cache

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.special import lambertw

from utils.array_definitions import ScanMap2DArray


@cache
def get_alpha(regression_order: int) -> float:
    """
    Return the appropriate alpha parameter value for the given regression order.

    :param regression_order: The order of the regression filter (can be 0, 1 or 2).
    :return: The alpha value corresponding to the given regression filter order.
    """
    if regression_order == 0 or regression_order == 1:
        return np.sqrt(np.log(2) / np.pi)
    elif regression_order == 2:
        w = lambertw(-1 / (2 * np.e), k=-1).real
        return np.sqrt((-1 - w) / np.pi)
    else:
        raise ValueError(f"Maximum regression order is 2, got {regression_order}")


def get_sigmas(
    alpha: float, cutoff_lengths: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Convert MATLAB-style cutoff lengths to equivalent scipy sigma values for Gaussian filters.

    The function translates the MATLAB Gaussian filter parameterization to the equivalent
    scipy Gaussian sigma. The formulas for the two filter representations are as follows:

    MATLAB Gaussian: 1/(alpha*cutoff) * exp(-pi * (x / (alpha * cutoff))^2)
    scipy Gaussian:  1/(sigma*sqrt(2*pi)) * exp(-x^2 / (2 * sigma^2))
    To match these two, the relationship between the MATLAB cutoff length and scipy sigma is:
    sigma = alpha * cutoff / sqrt(2 * pi)

    :param alpha: Smoothing parameter that controls the filter's frequency response.
    :param cutoff_lengths: Cutoff wavelength in pixels.
    :return: The standard deviation for a Gaussian kernel
    """
    return alpha * cutoff_lengths / np.sqrt(2 * np.pi)


def _apply_nan_weighted_filter(
    data: NDArray[np.floating],
    sigma: NDArray[np.floating],
    radius: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Apply Gaussian filter with NaN-aware weighting.

    NaN values are excluded from the convolution by setting their weight to 0.
    The result is normalized by the sum of weights to compensate.

    :param data: Input 2D data array (may contain NaN).
    :param sigma: The standard deviation used in the Gaussian kernel
    :param radius: Kernel radius for each axis.
    :return: Filtered data (NaN positions will have interpolated values).
    """
    gaussian_filter = partial(
        ndimage.gaussian_filter,
        sigma=sigma,
        mode="constant",
        cval=0,
        radius=radius,
    )

    nan_mask = np.isnan(data)
    filtered = gaussian_filter(np.where(nan_mask, 0, data))
    weight_sum = gaussian_filter((~nan_mask).astype(float))

    with np.errstate(invalid="ignore", divide="ignore"):
        return filtered / weight_sum


def get_cutoff_pixels(
    cutoff_lengths: tuple[float, float], pixel_size: tuple[float, float]
) -> NDArray[np.floating]:
    """
    Convert cutoff length from physical units to pixel units.

    :param cutoff_lengths: Cutoff wavelengths in physical units.
    :param pixel_size: Pixel size in physical units.
    :return: Cutoff wavelengths in pixel units.
    """
    return np.array(cutoff_lengths) / np.array(pixel_size)


def apply_gaussian_filter(
    data: ScanMap2DArray,
    cutoff_lengths: tuple[float, float],
    pixel_size: tuple[float, float] = (1.0, 1.0),
    regression_order: int = 0,
    nan_out: bool = True,
    is_high_pass: bool = False,
) -> NDArray[np.floating]:
    """
    Apply Gaussian filter to 2D data with NaN handling.

    :param data: Input 2D data array.
    :param cutoff_lengths: Cutoff wavelengths in physical units.
    :param pixel_size: The pixel size in meters (m).
    :param regression_order: Degree regression filter (0, 1 or 2).
    :param nan_out: If True, preserve NaN positions in output (zeros otherwise)
    :param is_high_pass: Whether to apply as highpass or lowpass filter (default)
    :return: Filtered data array.
    """
    if np.all(np.isnan(cutoff_lengths)):
        raise ValueError("All cutoff lengths are NaN, no filtering possible.")

    # Convert cutoff to pixel units and scipy parameters
    cutoff_pixels = get_cutoff_pixels(cutoff_lengths, pixel_size)
    sigma = get_sigmas(get_alpha(regression_order), cutoff_pixels)
    radius = np.ceil(cutoff_pixels).astype(int)

    # Weighted filtering for NaN handling
    filtered = _apply_nan_weighted_filter(data, sigma, radius)

    if nan_out:
        filtered[np.isnan(data)] = np.nan

    if is_high_pass:
        filtered = data - filtered

    return filtered
