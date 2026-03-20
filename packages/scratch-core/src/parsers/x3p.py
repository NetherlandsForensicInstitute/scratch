from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from x3p import X3Pfile
from x3p._x3pfileclasses import Ax

if TYPE_CHECKING:
    from container_models.scan_image import ScanImage


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


def _set_incremental_axis(axis: Ax, scale: float):
    axis.set_axistype("I")
    axis.set_increment(scale)
    axis.set_datatype("D")
    axis.set_offset(0.0)


def _set_record1_entries(x3p: X3Pfile, image: ScanImage) -> X3Pfile:
    """Set Record1 entries (axes configuration)."""
    x3p.record1.set_featuretype("SUR")
    _set_incremental_axis(x3p.record1.axes.CX, image.scale_x)
    _set_incremental_axis(x3p.record1.axes.CY, image.scale_y)
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


def convert_to_x3p(image: ScanImage) -> X3Pfile:
    """Convert ScanImage to X3Pfile."""
    file = X3Pfile()
    file = _set_record1_entries(file, image=image)
    file = _set_record2_entries(file, meta_data=X3PMetaData())
    file = _set_binary_data(file, image=image)
    return _set_record3_entries(file, image=image)


def save_x3p(x3p: X3Pfile, output_path: Path) -> None:
    """
    Save an X3P file to disk.

    :param x3p: The X3P file to save.
    :param output_path: The path where the file should be written.
    :returns: An ``IOResult[Path, Exception]`` — ``IOSuccess(Path)`` on success,
              or ``IOFailure(Exception)`` if an error occurs.
    """

    x3p.write(str(output_path))
