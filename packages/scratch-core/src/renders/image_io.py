import numpy as np

from container_models.base import FloatArray
from container_models.scan_image import ScanImage


def _clip_data(data: FloatArray, std_scaler: float) -> FloatArray:
    """
    Clip the data so that the values lie in the interval [μ - σ * S, μ + σ * S].

    Here the standard deviation σ is normalized by N-1. Note: NaN values are ignored and unaffected.

    :param data: The data to be clipped.
    :param std_scaler: The multiplier for the standard deviation of the data to be clipped.
    :returns: clipped data.
    """
    if std_scaler <= 0.0:
        raise ValueError("`std_scaler` must be a positive number.")
    mean = np.nanmean(data)
    std = np.nanstd(data, ddof=1) * std_scaler
    upper = float(mean + std)
    lower = float(mean - std)
    return np.clip(data, lower, upper)


def get_scan_image_for_display(
    scan_image: ScanImage, *, std_scaler: float = 2.0
) -> ScanImage:
    """
    Clip and normalize image data for displaying purposes.

    First the data will be clipped so that the values lie in the interval [μ - σ * S, μ + σ * S].
    Then the values are min-max normalized and scaled to the [0, 255] interval.

    :param scan_image: An instance of `ScanImage`.
    :param std_scaler: The multiplier `S` for the standard deviation used above when clipping the image.
    :returns: An array containing the clipped and normalized image data.
    """
    return ScanImage(
        data=_clip_data(data=scan_image.data, std_scaler=std_scaler),
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )
