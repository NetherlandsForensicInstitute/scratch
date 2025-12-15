from pathlib import Path

import numpy as np
from returns.io import impure_safe
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
def subsample_scan_image(
    scan_image: ScanImage, step_size: tuple[int, int]
) -> ScanImage:
    """
    Subsample the data in a `ScanImage` instance by skipping `step_size` steps.
    :param scan_image: The instance of `ScanImage` containing the 2D image data to subsample.
    :param step_size: Denotes the number of steps to skip in each dimension. The first integer
        corresponds to the subsampling step size in the X-direction, and the second integer to
        the step size in the Y-direction.
    """
    step_x, step_y = step_size
    width, height = scan_image.data.shape
    if step_x >= width or step_y >= height:
        raise ValueError("Step size should be smaller than the image size")
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Step size must be a tuple of positive integers")
    return ScanImage(
        data=scan_image.data[::step_y, ::step_x].copy(),
        scale_x=scan_image.scale_x * step_x,
        scale_y=scan_image.scale_y * step_y,
    )
