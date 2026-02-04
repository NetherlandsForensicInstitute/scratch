import datetime as dt
from typing import NamedTuple

import numpy as np
from returns.curry import partial
from returns.pipeline import pipe
from returns.result import safe
from x3p import X3Pfile
from x3p._x3pfileclasses import Ax

from container_models import ImageContainer
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


def _set_incremental_axis(axis: Ax, scale: float):
    axis.set_axistype("I")
    axis.set_increment(scale)
    axis.set_datatype("D")
    axis.set_offset(0.0)


def _set_record1_entries(x3p: X3Pfile, image: ImageContainer) -> X3Pfile:
    """Set Record1 entries (axes configuration)."""
    x3p.record1.set_featuretype("SUR")
    _set_incremental_axis(x3p.record1.axes.CX, image.metadata.scale.x)
    _set_incremental_axis(x3p.record1.axes.CY, image.metadata.scale.y)
    x3p.record1.axes.CZ.set_datatype("D")
    return x3p


def _set_record2_entries(x3p: X3Pfile, metadata: X3PMetaData) -> X3Pfile:
    """Set Record2 entries (metadata)."""
    x3p.record2.set_date(dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%S"))  # type: ignore
    x3p.record2.set_calibrationdate(metadata.calibration_date)  # type: ignore
    if metadata.author:
        x3p.record2.set_creator(metadata.author)  # type: ignore
    if metadata.comment:
        x3p.record2.set_comment(metadata.comment)  # type: ignore
    x3p.record2.instrument.set_model(metadata.model)  # type: ignore
    x3p.record2.instrument.set_manufacturer(metadata.manufacturer)  # type: ignore
    x3p.record2.instrument.set_version(metadata.instrument_version)  # type: ignore
    x3p.record2.probingsystem.set_identification(metadata.identificaton)  # type: ignore
    x3p.record2.probingsystem.set_type(metadata.measurement_type)  # type: ignore
    return x3p


def _set_binary_data(x3p: X3Pfile, image: ImageContainer) -> X3Pfile:
    """Set the binary data."""
    x3p.set_data(np.ascontiguousarray(image.data))
    return x3p


def _set_record3_entries(x3p: X3Pfile, image: ImageContainer) -> X3Pfile:
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
def parse_to_x3p(image: ImageContainer) -> X3Pfile:
    """Convert ImageContainer to X3Pfile using a functional approach."""
    return pipe(
        partial(_set_record1_entries, image=image),
        partial(_set_record2_entries, metadata=X3PMetaData()),
        partial(_set_binary_data, image=image),
        partial(_set_record3_entries, image=image),
    )(X3Pfile())
