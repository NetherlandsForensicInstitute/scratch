from container_models.scan_image import ScanImage
from scipy.ndimage import rotate
import numpy as np


def convert_meters_to_pixels(value_to_convert: float, pixel_size: float) -> int:
    """TODO: Remove this function if possible."""
    return int(round(value_to_convert / pixel_size))


def convert_pixels_to_meters(
    coordinates: tuple[int, int], pixel_size: tuple[float, float]
) -> tuple[float, float]:
    """TODO: Remove this function if possible."""
    return coordinates[0] * pixel_size[0], coordinates[1] * pixel_size[1]


def rotate_scan_image(scan_image: ScanImage, angle: float) -> ScanImage:
    """
    Rotate an instance of `ScanImage` by `angle` degrees.
    Background values are filled with NaNs.
    """
    return scan_image.model_copy(
        update={"data": rotate(scan_image.data, angle=angle, cval=np.nan)}
    )
