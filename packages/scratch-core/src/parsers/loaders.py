from pathlib import Path
from functools import lru_cache

import numpy as np
from returns.io import impure_safe
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC

from container_models import ImageContainer
from container_models.image import MetaData
from utils.logger import log_railway_function

from .patches.al3d import read_al3d
from scipy.constants import micro

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


@lru_cache(maxsize=1)
@log_railway_function(
    "Failed to load image file",
    "Successfully loaded scan file",
)
@impure_safe
def load_scan_image(scan_file: Path) -> ImageContainer:
    """
    Load a scan image from a file. Parsed values will be converted to meters (m).
    :param scan_file: The path to the file containing the scanned image data.
    :returns: An instance of `ImageContainer`.
    """
    surface = Surface.load(scan_file)

    return ImageContainer(
        data=np.asarray(surface.data, dtype=np.float64) * micro,
        metadata=MetaData.model_validate(
            surface.metadata
            | {"scale": (surface.step_x * micro, surface.step_y * micro)}
        ),
    )
