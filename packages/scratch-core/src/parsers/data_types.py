from pathlib import Path

import numpy as np
from returns.io import impure_safe
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC

from image_generation.data_formats import ScanImage
from parsers.exceptions import PreProcessError
from utils.logger import log_io_railway_function

from .patches.al3d import read_al3d
from scipy.constants import micro


# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


@log_io_railway_function(
    PreProcessError.SURFACE_LOAD_ERROR, "Successfully loaded scan file"
)
@impure_safe
def from_file(scan_file: Path) -> ScanImage:
    """
    Load a scan image from a file. Parsed values will be converted to meters (m).

    :param scan_file: The path to the file containing the scanned image data.
    :returns: An instance of `ScanImage`.
    """
    surface = Surface.load(scan_file)
    return ScanImage(
        data=np.asarray(surface.data, dtype=np.float64) * micro,
        scale_x=surface.step_x * micro,
        scale_y=surface.step_y * micro,
        meta_data=surface.metadata,
    )
