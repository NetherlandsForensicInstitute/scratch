import datetime as dt
from pathlib import Path

import numpy as np
from PIL import Image
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC
from x3p import X3Pfile

from .data_types import ParsedImage
from .patches.al3d import read_al3d

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


def parse_scan(path: Path) -> ParsedImage:
    """Parse surface data from a scan file."""
    surface = Surface.load(path)
    return ParsedImage(
        data=surface.data, scale_x=surface.step_x, scale_y=surface.step_y, metadata=surface.metadata, path=path
    )


def parse_file(path: Path) -> ParsedImage:
    """Parse surface data from an image file."""
    image = Image.open(path).convert("L")  # convert parsed image to grayscale
    data = np.asarray(image, dtype=np.float64) / 255.0  # scale pixel values to unit range
    return ParsedImage(data=data, path=path)


def _to_x3p(image: ParsedImage, author: str | None = None, comment: str | None = None) -> X3Pfile:
    x3p = X3Pfile()
    x3p.record1.set_featuretype("SUR")
    x3p.record1.axes.CX.set_axistype("I")
    x3p.record1.axes.CX.set_increment(image.scale_x)  # TODO: test if this should be in m or um
    x3p.record1.axes.CY.set_axistype("I")
    x3p.record1.axes.CY.set_increment(image.scale_y)  # TODO: test if this should be in m or um
    x3p.record2.set_date(dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%S"))
    if author:
        x3p.record2.set_creator(author)
    if comment:
        x3p.record2.set_comment(comment)
    # TODO: include more metadata in function arguments? (e.g. calibration date)
    x3p.record2.instrument.set_model("")
    x3p.record2.instrument.set_manufacturer("NFI")
    x3p.set_data(image.data)
    return x3p


def save_to_x3p(image: ParsedImage, path: Path, author: str | None = None, comment: str | None = None) -> None:
    """Save an instance of `ParsedImage` to a .x3p-file."""
    # TODO: extend function arguments to handle more meta-data fields?
    _to_x3p(image, author, comment).write(str(path))
