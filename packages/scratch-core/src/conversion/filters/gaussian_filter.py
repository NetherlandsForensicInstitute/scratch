from functools import partial
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.special import lambertw

from .protocol import FilterFlags
from utils.array_definitions import ScanMap2DArray


def get_alpha(regression_order: int) -> float:
    """
    Return the appropriate alpha parameter value for the given regression order.

    :param regression_order: The order of the regression filter.
    :return: The alpha value corresponding to the given regression filter order.
    """
    if regression_order <= 1:
        return np.sqrt(np.log(2) / np.pi)
    if regression_order == 2:
        w = lambertw(-1 / (2 * np.e), k=-1).real
        return np.sqrt((-1 - w) / np.pi)
    raise ValueError(f"Maximum regression order is 2, got {regression_order}")


def get_cutoff_sigmas(
    alpha: float, cutoff_lengths: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Convert MATLAB-style cutoff length to scipy sigma.

    MATLAB Gaussian: 1/(alpha*cutoff) * exp(-pi * (x / (alpha * cutoff))^2)
    scipy Gaussian:  1/(sigma*sqrt(2*pi)) * exp(-x^2 / (2 * sigma^2))

    Matching: sigma = alpha * cutoff / sqrt(2 * pi)

    :param alpha: Alpha value
    :param cutoff_length: Cutoff wavelength in pixels.
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


def apply_gaussian_filter(
    data: ScanMap2DArray,
    alpha: float,
    *,
    cutoff_pixels: NDArray[np.floating],
    flags: FilterFlags = FilterFlags.NAN_OUT & ~FilterFlags.HIGH_PASS,
) -> np.ndarray:
    # Weighted filtering for NaN handling
    filtered = _apply_nan_weighted_filter(
        data,
        sigma=get_cutoff_sigmas(alpha, cutoff_pixels),
        radius=np.ceil(cutoff_pixels).astype(int),
    )

    if FilterFlags.NAN_OUT in flags:
        filtered[np.isnan(data)] = np.nan

    if FilterFlags.HIGH_PASS in flags:
        return data - filtered

    return filtered
