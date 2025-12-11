from pathlib import Path

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


# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


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
    "Failed to subsample image file",
    "Successfully subsampled scan file",
)
@safe
def subsample_scan_image(
    scan_image: ScanImage, step_size_x: int, step_size_y: int
) -> ScanImage:
    """
    Subsample the data in a `ScanImage` instance by skipping steps in each dimension.
    :param scan_image: The instance of `ScanImage` containing the 2D image data to subsample.
    :param step_size_x: The number of steps to skip in the X-direction.
    :param step_size_y: The number of steps to skip in the Y-direction.
    :returns: An subsampled `ScanImage` with updated scales.
    """
    width, height = scan_image.data.shape
    if not (0 < step_size_x < width and 0 < step_size_y < height):
        raise ValueError(
            f"Step size should be positive and smaller than the image size: {(height, width)}"
        )
    return ScanImage(
        data=scan_image.data[::step_size_y, ::step_size_x].copy(),
        scale_x=scan_image.scale_x * step_size_x,
        scale_y=scan_image.scale_y * step_size_y,
    )
