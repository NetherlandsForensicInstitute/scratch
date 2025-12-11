import datetime as dt
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
from returns.pipeline import flow
from x3p import X3Pfile
from returns.result import safe
from returns.io import impure_safe
from container_models.scan_image import ScanImage
from utils.logger import log_railway_function


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


def _set_record1_entries(x3p: X3Pfile, image: ScanImage) -> X3Pfile:
    """Set Record1 entries (axes configuration)."""
    x3p.record1.set_featuretype("SUR")
    x3p.record1.axes.CX.set_axistype("I")
    x3p.record1.axes.CX.set_increment(image.scale_x)
    x3p.record1.axes.CX.set_datatype("D")
    x3p.record1.axes.CY.set_axistype("I")
    x3p.record1.axes.CY.set_increment(image.scale_y)
    x3p.record1.axes.CY.set_datatype("D")
    x3p.record1.axes.CZ.set_datatype("D")
    return x3p


def _set_record2_entries(x3p: X3Pfile, meta_data: X3PMetaData) -> X3Pfile:
    """Set Record2 entries (metadata)."""
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
    return x3p


def _set_binary_data(x3p: X3Pfile, image: ScanImage) -> X3Pfile:
    """Set the binary data."""
    x3p.set_data(np.ascontiguousarray(image.data))
    return x3p


def _set_record3_entries(x3p: X3Pfile, image: ScanImage) -> X3Pfile:
    """Set Record3 entries (matrix dimensions)."""
    # manually set the Record3 entries since these are set incorrectly in package
    x3p.record3.matrixdimension.sizeX = image.data.shape[1]  # type: ignore
    x3p.record3.matrixdimension.sizeY = image.data.shape[0]  # type: ignore
    return x3p


@log_railway_function(
    "Failed to parse image X3P",
    "Successfully parse array to x3p",
)
@safe
def parse_to_x3p(image: ScanImage) -> X3Pfile:
    """Convert ScanImage to X3Pfile using a functional approach."""
    return flow(
        X3Pfile(),
        partial(_set_record1_entries, image=image),
        partial(_set_record2_entries, meta_data=X3PMetaData()),
        partial(_set_binary_data, image=image),
        partial(_set_record3_entries, image=image),
    )


@log_railway_function(
    "Failed to write X3P file",
    "Successfully written X3P",
)
@impure_safe
def save_x3p(x3p: X3Pfile, output_path: Path) -> Path:
    """Save an X3Pfile to disk.

    Args:
        x3p: The X3Pfile to save
        output_path: Where to save the file

    Returns:
        IOResult[Path, Exception]: IOSuccess(Path) on success, IOFailure(Exception) on error
    """

    x3p.write(str(output_path))
    return output_path
