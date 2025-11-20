from parsers.data_types import ScanImage, Array2D
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage


def clip_data(data: NDArray, std_scaler) -> tuple[NDArray, float, float]:
    """
    Clip the data so that the values lie in the interval [μ - σ * S, μ + σ * S].

    Here the standard deviation σ is normalized by N-1. Note: NaN values are ignored and unaffected.

    :param data: The data to be clipped.
    :param std_scaler: The multiplier for the standard deviation of the data to be clipped.
    :returns: A tuple containing the clipped data, the lower bound, and the upper bound of the clipped data.
    """
    mean = np.nanmean(data)
    std = np.nanstd(data, ddof=1) * std_scaler
    upper = float(mean + std)
    lower = float(mean - std)
    clipped = np.clip(data, lower, upper)
    return clipped, lower, upper


def get_image_for_display(
    image: Array2D | ScanImage, std_scaler: float = 2.0
) -> PILImage:
    """
    Get an 8-bit RGBA image for displaying a scan image.

    First the data will be clipped and normalized so that the values lie in the interval
    [μ - σ * S, μ + σ * S]. Then the values are min-max normalized and scaled to the [0, 255] interval
    before they are converted to 8-bit unsigned integers. NaN values will be converted to black pixels
    with 100% transparency.

    :param image: Either a 2D NumPy array containing the raw scan data or an instance of `ScanImage`.
    :param std_scaler: The multiplier `S` for the standard deviation used above when clipping the image.
    :returns: A PIL Image object in 8-bit RGBA format.
    """
    if isinstance(image, ScanImage):
        image = image.data
    clipped, lower, upper = clip_data(data=image, std_scaler=std_scaler)
    normalized = (clipped - lower) / (upper - lower) * 255.0
    rgba = np.empty(shape=(*image.shape, 4), dtype=np.uint8)
    rgba[..., :-1] = np.expand_dims(normalized.astype(np.uint8), axis=-1)
    rgba[..., -1] = ~np.isnan(normalized) * 255
    return Image.fromarray(rgba)
