import datetime as dt
from pathlib import Path
from typing import NamedTuple

import numpy as np
from x3p import X3Pfile

from .data_types import ScanImage


class X3PMetaData(NamedTuple):
    """A container for storing X3P meta-data."""

    # TODO: parse default values from a config file?
    author: str | None = None
    comment: str | None = None
    instrument_version: str | None = None
    identificaton: str | None = None
    calibration_date: str | None = None
    model: str = "Default"
    manufacturer: str = "NFI"
    measurement_type: str = "NonContacting"


def _to_x3p(image: ScanImage, meta_data: X3PMetaData) -> X3Pfile:
    x3p = X3Pfile()
    # set record1 entries
    x3p.record1.set_featuretype("SUR")
    x3p.record1.axes.CX.set_axistype("I")
    x3p.record1.axes.CX.set_increment(image.scale_x)
    x3p.record1.axes.CX.set_datatype("D")
    x3p.record1.axes.CY.set_axistype("I")
    x3p.record1.axes.CY.set_increment(image.scale_y)
    x3p.record1.axes.CY.set_datatype("D")
    x3p.record1.axes.CZ.set_datatype("D")
    # set record2 entries
    x3p.record2.set_date(dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%S"))  # type: ignore
    x3p.record2.set_calibrationdate(meta_data.calibration_date)  # type: ignore
    if meta_data.author:
        x3p.record2.set_creator(meta_data.author)  # type: ignore
    if meta_data.comment:
        x3p.record2.set_comment(meta_data.comment)  # type: ignore
    x3p.record2.instrument.set_model(meta_data.model)  # type: ignore
    x3p.record2.instrument.set_manufacturer(meta_data.manufacturer)  # type: ignore
    x3p.record2.instrument.set_version(meta_data.instrument_version)  # type: ignore
    x3p.record2.probingsystem.set_identification(meta_data.identificaton)  # type: ignore
    x3p.record2.probingsystem.set_type(meta_data.measurement_type)  # type: ignore
    # set the binary data
    x3p.set_data(np.ascontiguousarray(image.data))
    # manually set the Record3 entries since these are set incorrectly in package
    x3p.record3.matrixdimension.sizeX = image.data.shape[1]
    x3p.record3.matrixdimension.sizeY = image.data.shape[0]
    return x3p


def save_to_x3p(
    image: ScanImage, output_path: Path, meta_data: X3PMetaData | None = None
) -> None:
    """Save an instance of `ScanImage` to a .x3p-file."""
    _to_x3p(image, meta_data or X3PMetaData()).write(str(output_path))
