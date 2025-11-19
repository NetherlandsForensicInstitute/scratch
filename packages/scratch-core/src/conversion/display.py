from parsers.data_types import Array2D, ScanImage
import numpy as np
from numpy.typing import DTypeLike, NDArray


def clip_data(data: NDArray, std_scaler: float = 2.0) -> tuple[NDArray, float, float]:
    """
    Clip the data so that the values lie in the interval [μ - σ * std_scaler, μ + σ * std_scaler].

    :param data: The data to be clipped.
    :param std_scaler: The multiplier for the standard deviation of the data to be clipped.
    :returns: A tuple containing the clipped data, the lower bound, and the upper bound of the clipped data.
    """
    std = np.nanstd(data) * std_scaler
    mean = np.nanmean(data)
    upper = float(mean + std)
    lower = float(mean - std)
    clipped = np.clip(data, lower, upper)
    return clipped, lower, upper


def get_image_for_display(scan_image: ScanImage, std_scaler: float = 2.0) -> Array2D:
    """
    Get a 2D image from the scan data for displaying purposes.

    The scan data will be clipped and normalized, and converted to
    """
    data = scan_image.data
    clipped, lower, upper = clip_data(data=data, std_scaler=std_scaler)
    normalized = (clipped - lower) / (upper - lower) * np.iinfo(dtype).max
    return normalized.astype(dtype)
