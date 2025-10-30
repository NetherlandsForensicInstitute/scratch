import datetime as dt
from pathlib import Path
from typing import NamedTuple

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


class X3PMetaData(NamedTuple):
    author: str | None = None
    comment: str | None = None
    instrument_version: str | None = None
    identificaton: str | None = None
    calibration_date: str | None = None
    model: str = "Default"
    manufacturer: str = "NFI"
    measurement_type: str = "NonContacting"


def parse_x3p(path: Path) -> ParsedImage:
    """Parse surface data from a x3p file."""
    x3p = X3Pfile(str(path))
    return ParsedImage(
        data=np.asarray(x3p.data, dtype=np.float64),
        scale_x=x3p.record1.axes.CX.increment,
        scale_y=x3p.record1.axes.CX.increment,
        path_to_original_image=path,
    )


def parse_scan(path: Path) -> ParsedImage:
    """Parse surface data from a scan file."""
    surface = Surface.load(path)
    surface.data = np.asarray(surface.data, dtype=np.float64)
    return ParsedImage(
        data=surface.data,
        scale_x=surface.step_x,
        scale_y=surface.step_y,
        metadata=surface.metadata,
        path_to_original_image=path,
    )


def parse_image(path: Path) -> ParsedImage:
    """Parse surface data from an image file."""
    # open image and convert to grayscale
    image = Image.open(path).convert("L")
    data = np.asarray(image, dtype=np.float64)
    return ParsedImage(data=data, path_to_original_image=path)


def _to_x3p(image: ParsedImage, meta_data: X3PMetaData) -> X3Pfile:
    x3p = X3Pfile()
    # write record1 entries
    x3p.record1.set_featuretype("SUR")
    x3p.record1.axes.CX.set_axistype("I")
    x3p.record1.axes.CX.set_increment(image.scale_x)
    x3p.record1.axes.CY.set_axistype("I")
    x3p.record1.axes.CY.set_increment(image.scale_y)
    # write record2 entries
    x3p.record2.set_date(dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%S"))
    x3p.record2.set_calibrationdate(meta_data.calibration_date)
    if meta_data.author:
        x3p.record2.set_creator(meta_data.author)
    if meta_data.comment:
        x3p.record2.set_comment(meta_data.comment)
    x3p.record2.instrument.set_model(meta_data.model)
    x3p.record2.instrument.set_manufacturer(meta_data.manufacturer)
    x3p.record2.instrument.set_version(meta_data.instrument_version)
    x3p.record2.probingsystem.set_type(meta_data.measurement_type)
    # write the binary data
    x3p.set_data(np.ascontiguousarray(image.data))

    return x3p


def save_to_x3p(image: ParsedImage, path: Path, meta_data: X3PMetaData | None = None) -> None:
    """Save an instance of `ParsedImage` to a .x3p-file."""
    _to_x3p(image, meta_data or X3PMetaData()).write(str(path))


FILETYPE_TO_PARSER = {
    ".png": parse_image,
    ".al3d": parse_scan,
    ".x3p": parse_x3p,
    # TODO: add more file types
}


def parse_file(path: Path) -> ParsedImage:
    """Parse a surface scan file and return an instance of `ParsedImage`."""
    parser = FILETYPE_TO_PARSER.get(path.suffix.lower())
    if parser:
        return parser(path)
    else:
        raise RuntimeError(f"File type not supported: {path.suffix}")
