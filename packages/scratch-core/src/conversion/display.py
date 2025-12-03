import numpy as np

from conversion.exceptions import NegativeStdScalerException
from utils.array_definitions import ScanMap2DArray, ScanMapRGBA


def clip_data(
    data: ScanMap2DArray, std_scaler: float
) -> tuple[ScanMap2DArray, float, float]:
    """
    Clip the data so that the values lie in the interval [μ - σ * S, μ + σ * S].

    Here the standard deviation σ is normalized by N-1. Note: NaN values are ignored and unaffected.

    :param data: The data to be clipped.
    :param std_scaler: The multiplier for the standard deviation of the data to be clipped.
    :returns: A tuple containing the clipped data, the lower bound, and the upper bound of the clipped data.
    """
    if std_scaler <= 0.0:
        raise NegativeStdScalerException("`std_scaler` must be a positive number.")
    mean = np.nanmean(data)
    std = np.nanstd(data, ddof=1) * std_scaler
    upper = float(mean + std)
    lower = float(mean - std)
    clipped = np.clip(data, lower, upper)
    return clipped, lower, upper


def grayscale_to_rgba(image: ScanMap2DArray) -> ScanMapRGBA:
    """
    Convert a 2D grayscale array to an 8-bit RGBA array.

    The grayscale pixel values are assumed to be floating point values in the [0, 255] interval.
    NaN values will be converted to black pixels with 100% transparency.

    :param image: The grayscale image data to be converted to an 8-bit RGBA image.
    :returns: Array with the image data in 8-bit RGBA format.
    """
    rgba = np.empty(shape=(*image.shape, 4), dtype=np.uint8)
    rgba[..., :-1] = np.expand_dims(image.astype(np.uint8), axis=-1)
    rgba[..., -1] = ~np.isnan(image) * 255
    return rgba


def normalize(
    input_array: ScanMap2DArray, lower: float, upper: float
) -> ScanMap2DArray:
    """Perform min-max normalization on the input array and scale to the [0, 255] interval."""
    if lower >= upper:
        raise ValueError(
            f"The lower bound ({lower}) should be smaller than the upper bound ({upper})."
        )
    return (input_array - lower) / (upper - lower) * 255.0
