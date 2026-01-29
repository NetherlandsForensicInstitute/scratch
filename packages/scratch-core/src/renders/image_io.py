from pathlib import Path
from PIL.Image import Image, fromarray
import numpy as np

from returns.io import impure_safe
from returns.result import safe

from container_models.scan_image import ScanImage
from container_models.base import ImageRGBA, FloatArray2D, FloatArray
from utils.logger import log_railway_function


def grayscale_to_rgba(scan_data: FloatArray2D) -> ImageRGBA:
    """
      Convert a 2D grayscale array to an 8-bit RGBA array.

    The grayscale pixel values are assumed to be floating point values in the [0, 255] interval.
    NaN values will be converted to black pixels with 100% transparency.

    :param scan_data: The grayscale image data to be converted to an 8-bit RGBA image.
    :returns: Array with the image data in 8-bit RGBA format.
    """
    gray_uint8 = np.nan_to_num(scan_data, nan=0.0).astype(np.uint8)
    rgba = np.repeat(gray_uint8[..., np.newaxis], 4, axis=-1)
    rgba[..., 3] = (~np.isnan(scan_data)).astype(np.uint8) * 255
    return rgba


def _normalize(input_array: FloatArray, lower: float, upper: float) -> FloatArray:
    """Perform min-max normalization on the input array and scale to the [0, 255] interval."""
    if lower >= upper:
        raise ValueError(
            f"The lower bound ({lower}) should be smaller than the upper bound ({upper})."
        )
    return (input_array - lower) / (upper - lower) * 255.0


def _clip_data(data: FloatArray, std_scaler: float) -> tuple[FloatArray, float, float]:
    """
    Clip the data so that the values lie in the interval [μ - σ * S, μ + σ * S].

    Here the standard deviation σ is normalized by N-1. Note: NaN values are ignored and unaffected.

    :param data: The data to be clipped.
    :param std_scaler: The multiplier for the standard deviation of the data to be clipped.
    :returns: A tuple containing the clipped data, the lower bound, and the upper bound of the clipped data.
    """
    if std_scaler <= 0.0:
        raise ValueError("`std_scaler` must be a positive number.")
    mean = np.nanmean(data)
    std = np.nanstd(data, ddof=1) * std_scaler
    upper = float(mean + std)
    lower = float(mean - std)
    return np.clip(data, lower, upper), lower, upper


@log_railway_function("Failed to retrieve array for display")
@safe
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
        data=_normalize(*_clip_data(data=scan_image.data, std_scaler=std_scaler)),
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )


@log_railway_function("Failed to convert scan to image")
@safe
def scan_to_image(scan_image: ScanImage) -> Image:
    return fromarray(grayscale_to_rgba(scan_data=scan_image.data))


@log_railway_function("Failed to convert grayscale data to image")
@safe
def grayscale_to_image(grayscale: FloatArray2D) -> Image:
    return fromarray(grayscale_to_rgba(scan_data=grayscale))


@log_railway_function("Failed to save image")
@impure_safe
def save_image(image: Image, output_path: Path) -> Path:
    image.save(output_path)
    return output_path
