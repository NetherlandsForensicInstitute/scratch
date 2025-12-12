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
def load_scan_image(
    scan_file: Path, step_size_x: int = 1, step_size_y: int = 1
) -> ScanImage:
    """
    Load a scan image from a file and optionally subsample it. Parsed values will be converted to meters (m).

    :param scan_file: The path to the file containing the scanned image data.
    :param step_size_x: Denotes the number of steps to skip in X-direction (default: 1, no subsampling).
    :param step_size_y: Denotes the number of steps to skip in Y-direction (default: 1, no subsampling).
    :returns: An instance of `ScanImage`, optionally subsampled.
    """
    surface = Surface.load(scan_file)
    data = np.asarray(surface.data, dtype=np.float64) * micro
    step_x = surface.step_x * micro
    step_y = surface.step_y * micro
    height, width = data.shape

    if not (0 < step_size_x < width and 0 < step_size_y < height):
        raise ValueError(
            f"Step size should be positive and smaller than the image size: {(height, width)}"
        )

    if step_size_x > 1 or step_size_y > 1:
        return ScanImage(
            data=data[::step_size_y, ::step_size_x],
            scale_x=step_x * step_size_x,
            scale_y=step_y * step_size_y,
            meta_data=surface.metadata,
        )

    return ScanImage(
        data=data, scale_x=step_x, scale_y=step_y, meta_data=surface.metadata
    )
