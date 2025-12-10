from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.special import lambertw

from utils.array_definitions import ScanMap2DArray


def get_alpha(regression_order: int) -> float:
    """
    Return the appropriate alpha parameter value for the given regression order.

    :param regression_order: The order of the regression filter.
    :return: The alpha value corresponding to the given regression filter order.
    """
    if regression_order <= 1:
        return np.sqrt(np.log(2) / np.pi)
    elif regression_order == 2:
        w = lambertw(-1 / (2 * np.e), k=-1).real
        return np.sqrt((-1 - w) / np.pi)
    else:
        raise ValueError(f"Maximum regression order is 2, got {regression_order}")


def _apply_nan_weighted_filter(
    data: NDArray[np.floating],
    sigma: NDArray[np.floating],
    radius: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply Gaussian filter with NaN-aware weighting.

    NaN values are excluded from the convolution by setting their weight to 0.
    The result is normalized by the sum of weights to compensate.

    :param data: Input 2D data array (may contain NaN).
    :param sigma: Gaussian sigma for each axis.
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

    filtered = gaussian_filter(np.where(np.isnan(data), 0, data))
    weight_sum = gaussian_filter((~np.isnan(data)).astype(float))

    with np.errstate(invalid="ignore", divide="ignore"):
        return filtered / weight_sum


def get_cutoff_sigmas(
    alpha: float, cutoff_lengths: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Convert MATLAB-style cutoff length to scipy sigma.

    MATLAB Gaussian: 1/(alpha*cutoff) * exp(-pi * (x / (alpha * cutoff))^2)
    scipy Gaussian:  1/(sigma*sqrt(2*pi)) * exp(-x^2 / (2 * sigma^2))
    Matching: sigma = alpha * cutoff / sqrt(2 * pi)

    :param alpha: Alpha value
    :param cutoff_lengths: Cutoff wavelengths in pixels.
    :return: Equivalent scipy sigma.
    """
    return alpha * cutoff_lengths / np.sqrt(2 * np.pi)


def get_cutoff_pixels(
    cutoff_length: tuple[float, float], pixel_size: tuple[float, float]
) -> NDArray[np.floating]:
    """Convert cutoff length from physical units to pixel units.
    :param cutoff_length: Cutoff wavelength (row, col) in physical units.
    :param pixel_size: Pixel size (row, col) in physical units.
    :return: Cutoff wavelength (row, col) in pixel units.
    """
    return np.array(cutoff_length) / np.array(pixel_size)


def apply_gaussian_filter(
    data: ScanMap2DArray,
    cutoff_length: tuple[float, float],
    pixel_size: tuple[float, float] = (1.0, 1.0),
    regression_order: int = 0,
    nan_out: bool = True,
    is_high_pass: bool = False,
) -> np.ndarray:
    """Apply Gaussian filter to 2D data with NaN handling.

    :param data: Input 2D data array.
    :param cutoff_length: Cutoff wavelength (row, col) in physical units.
    :param pixel_size: The pixel size in the X-direction en Y-direction in meters (m).
    :param regression_order: Degree regression filter.
    :param nan_out: If True, preserve NaN positions in output.
    :param is_high_pass: Whether to apply as highpass filter (data - filtered).
    :return: Filtered data array.
    """
    if np.all(np.isnan(cutoff_length)):
        return data

    # Convert cutoff to pixel units and scipy parameters
    cutoff_pixels = get_cutoff_pixels(cutoff_length, pixel_size)
    sigma = get_cutoff_sigmas(get_alpha(regression_order), cutoff_pixels)
    radius = np.ceil(cutoff_pixels).astype(int)

    # Weighted filtering for NaN handling
    result = _apply_nan_weighted_filter(data, sigma, radius)

    if nan_out:
        result[np.isnan(data)] = np.nan

    if is_high_pass:
        result = data - result

    return result
