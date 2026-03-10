from container_models.scan_image import ScanImage
from scipy.ndimage import rotate
import numpy as np


def convert_meters_to_pixels(
    values: tuple[float, float], pixel_size: float
) -> tuple[int, int]:
    """TODO: Remove this function if possible."""

    def _convert(value: float) -> int:
        return int(round(value / pixel_size))

    return _convert(values[0]), _convert(values[1])


def convert_pixels_to_meters(
    values: tuple[float, float], pixel_size: float
) -> tuple[float, float]:
    """TODO: Remove this function if possible."""

    def _convert(value: float) -> float:
        return value * pixel_size

    return _convert(values[0]), _convert(values[1])


def rotate_scan_image(scan_image: ScanImage, angle: float) -> ScanImage:
    """
    Rotate an instance of `ScanImage` by `angle` degrees.
    Background values are filled with NaNs.
    """
    return scan_image.model_copy(
        update={"data": rotate(scan_image.data, angle=angle, cval=np.nan)}
    )
