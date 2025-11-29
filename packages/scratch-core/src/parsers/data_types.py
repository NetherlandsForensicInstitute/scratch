from pathlib import Path

import numpy as np
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC

from image_generation.data_formats import ScanMap2D

from .patches.al3d import read_al3d

UNIT_CONVERSION_FACTOR = 1e-6  # conversion factor from micrometers (um) to meters (m)

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


def from_file(scan_file: Path) -> "ScanMap2D":
    """
    Load a scan image from a file. Parsed values will be converted to meters (m).

    :param scan_file: The path to the file containing the scanned image data.
    :returns: An instance of `ScanMap2D`.
    """
    surface = Surface.load(scan_file)
    return ScanMap2D(
        data=np.asarray(surface.data, dtype=np.float64) * UNIT_CONVERSION_FACTOR,
        scale_x=surface.step_x * UNIT_CONVERSION_FACTOR,
        scale_y=surface.step_y * UNIT_CONVERSION_FACTOR,
        meta_data=surface.metadata,
    )
