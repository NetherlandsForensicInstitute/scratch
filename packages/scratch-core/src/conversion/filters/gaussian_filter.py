import numpy as np
from scipy import ndimage

from utils.array_definitions import ScanMap2DArray


def get_alpha(n_order):
    """Return the appropriate alpha value for the given degree regression filter."""
    if n_order <= 1:
        return np.sqrt(np.log(2) / np.pi)
    else:
        return 7.309134280946760e-01


def _cutoff_to_sigma(alpha: float, cutoff_length: float) -> float:
    """Convert MATLAB-style cutoff length to scipy sigma.

    MATLAB Gaussian: 1/(alpha*cutoff) * exp(-pi * (x / (alpha * cutoff))^2)
    scipy Gaussian:  1/(sigma*sqrt(2*pi)) * exp(-x^2 / (2 * sigma^2))

    Matching: sigma = alpha * cutoff / sqrt(2 * pi)

    :param alpha: Alpha value
    :param cutoff_length: Cutoff wavelength in pixels.
    :return: Equivalent scipy sigma.
    """
    return alpha * cutoff_length / np.sqrt(2 * np.pi)


def _cutoff_to_truncate(cutoff_length: float, sigma: float) -> float:
    """Calculate scipy truncate parameter to match MATLAB kernel size.

    MATLAB kernel radius = ceil(cutoff)
    scipy kernel radius = ceil(truncate * sigma)

    :param cutoff_length: Cutoff wavelength in pixels.
    :param sigma: scipy sigma parameter.
    :return: truncate parameter for scipy.
    """
    return np.ceil(cutoff_length) / sigma


def apply_gaussian_filter(
    data: ScanMap2DArray,
    cutoff_length: tuple[float, float],
    pixel_size: tuple[float, float] = (1.0, 1.0),
    n_order: int = 0,
    nan_out: bool = True,
) -> np.ndarray:
    """Apply Gaussian filter to 2D data with NaN handling.

    :param data: Input 2D data array.
    :param cutoff_length: Cutoff wavelength (row, col) in physical units.
    :param pixel_size: The pixel size in the X-direction en Y-direction in meters (m).
    :param n_order: Degree regression filter.
    :param nan_out: If True, preserve NaN positions in output.
    :return: Filtered data array.
    """
    # Convert cutoff to pixel units and scipy parameters
    cutoff_pixels = np.array(cutoff_length) / np.array(pixel_size)
    sigma = np.array([_cutoff_to_sigma(get_alpha(n_order), c) for c in cutoff_pixels])
    radius = np.ceil(cutoff_pixels).astype(int)

    # Weighted filtering for NaN handling
    weights = (~np.isnan(data)).astype(float)
    data_clean = np.where(np.isnan(data), 0, data)

    filtered = ndimage.gaussian_filter(
        data_clean, sigma=sigma, mode="constant", cval=0.0, radius=radius
    )
    weight_sum = ndimage.gaussian_filter(
        weights, sigma=sigma, mode="constant", cval=0.0, radius=radius
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        result = filtered / weight_sum

    if nan_out:
        result[np.isnan(data)] = np.nan

    return result
