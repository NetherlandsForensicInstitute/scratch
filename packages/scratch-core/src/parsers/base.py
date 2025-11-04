from pathlib import Path

import numpy as np
from PIL import Image
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC

from .data_types import ParsedImage
from .patches.al3d import read_al3d

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


def parse_scan(path: Path) -> ParsedImage:
    """
    Parse surface data from a raw scan file.

    :param path: Path to the raw scan file.
    :return: An instance of ParsedImage containing the parsed scan data.
    """
    surface = Surface.load(path)
    surface.data = np.asarray(surface.data, dtype=np.float64)
    return ParsedImage(
        data=surface.data,
        scale_x=surface.step_x,
        scale_y=surface.step_y,
        meta_data=surface.metadata,
        path_to_original_image=path,
    )


def parse_image(path: Path) -> ParsedImage:
    """
    Parse surface data from an image file.

    The parsed image data will be converted to grayscale and converted to floating point values.

    :param path: Path to the image file.
    :return: An instance of ParsedImage containing the parsed image data.
    """
    image = Image.open(path).convert("L")
    data = np.asarray(image, dtype=np.float64)
    return ParsedImage(data=data, path_to_original_image=path)
