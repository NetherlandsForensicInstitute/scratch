from pathlib import Path
from functools import lru_cache
from loguru import logger

from numpy.typing import NDArray
import numpy as np
from returns.io import impure_safe
from returns.result import safe
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC

from container_models.scan_image import ScanImage
from utils.logger import log_railway_function

from .patches.al3d import read_al3d
from scipy.constants import micro
from skimage.transform import resize

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


@lru_cache(maxsize=1)
@log_railway_function(
    "Failed to load image file",
    "Successfully loaded scan file",
)
@impure_safe
def load_scan_image(scan_file: Path) -> ScanImage:
    """
    Load a scan image from a file. Parsed values will be converted to meters (m).
    :param scan_file: The path to the file containing the scanned image data.
    :returns: An instance of `ScanImage`.
    """
    surface = Surface.load(scan_file)
    data = np.asarray(surface.data, dtype=np.float64) * micro
    step_x = surface.step_x * micro
    step_y = surface.step_y * micro

    return ScanImage(
        data=data,
        scale_x=step_x,
        scale_y=step_y,
        meta_data=surface.metadata,
    )


@log_railway_function(
    "Failed to make image resolution isotropic",
    "Successfully upsampled image file to isotropic resolution",
)
@safe
def make_isotropic(scan_image: ScanImage) -> ScanImage:
    """
    Resample a scan image to isotropic resolution (i.e. equal pixel spacing in X and Y).

    If the scan image is already isotropic, the original instance is returned.
    Otherwise, the image data is upsampled to the highest available resolution
    (the smaller of the two scale factors) using nearest-neighbor interpolation.
    Note: NaN values are preserved and will not be interpolated.

    :param scan_image: The ScanImage instance to be resampled.
    :returns: A new ScanImage instance with isotropic scaling.
    """
    if _is_isotropic(scan_image):
        return scan_image

    # Upsample to the smallest pixel scale (highest resolution)
    target_scale = min(scan_image.scale_x, scan_image.scale_y)

    return ScanImage(
        data=_upsample_image_data(scan_image, target_scale),
        scale_x=target_scale,
        scale_y=target_scale,
        meta_data=scan_image.meta_data,
    )


@log_railway_function(
    "Failed to subsample image file",
    "Successfully subsampled scan file",
)
@log_railway_function("Failed to subsample image file")
@safe
def subsample_scan_image(
    scan_image: ScanImage, step_size_x: int, step_size_y: int
) -> ScanImage:
    """
    Subsample the data in a `ScanImage` instance by skipping steps in each dimension.
    :param scan_image: The instance of `ScanImage` containing the 2D image data to subsample.
    :param step_size_x: The number of steps to skip in the X-direction.
    :param step_size_y: The number of steps to skip in the Y-direction.
    :returns: A subsampled `ScanImage` with updated scales. If both step sizes are 1, the original
        `ScanImage` instance is returned.
    """
    if step_size_x == 1 and step_size_y == 1:
        logger.info("No subsampling needed, returning original scan image")
        return scan_image

    width, height = scan_image.width, scan_image.height
    if not (0 < step_size_x < width and 0 < step_size_y < height):
        raise ValueError(
            f"Step size should be positive and smaller than the image size: {(height, width)}"
        )
    logger.info(
        "Subsampling scan image with step sizes x: {step_size_x}, y: {step_size_y}"
    )
    return ScanImage(
        data=scan_image.data[::step_size_y, ::step_size_x].copy(),
        scale_x=scan_image.scale_x * step_size_x,
        scale_y=scan_image.scale_y * step_size_y,
    )


def _is_isotropic(scan_image: ScanImage) -> bool:
    """Check if a scan image is isotropic within tolerance."""
    tolerance = 1e-16
    return bool(np.isclose(scan_image.scale_x, scan_image.scale_y, atol=tolerance))


def _get_target_shape(scan_image: ScanImage, target_scale: float) -> tuple[int, int]:
    """Get the target shape for a scan image given a target scale."""
    height, width = (
        int(round(scan_image.height * scan_image.scale_y / target_scale)),
        int(round(scan_image.width * scan_image.scale_x / target_scale)),
    )
    return height, width


def _upsample_image_data(
    scan_image: ScanImage, target_scale: float
) -> NDArray[np.float64]:
    """Upsample image data in a `ScanImage` instance to a common target scale."""
    upsampled = resize(
        image=scan_image.data,
        output_shape=_get_target_shape(scan_image, target_scale),
        mode="edge",
        anti_aliasing=False,  # Disabled for pure upsampling
        preserve_range=True,  # Keep original data intensity levels
        order=0,  # Nearest Neighbor so that NaNs appear at corresponding coordinates
    )
    return np.asarray(upsampled, dtype=np.float64)
