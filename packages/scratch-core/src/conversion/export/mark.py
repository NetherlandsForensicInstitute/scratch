"""
Module for serializing and deserializing Mark objects to/from disk.

This module provides functionality to save Mark objects as JSON metadata
with NPZ binary data, and load them back into memory.
"""

from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import BeforeValidator, Field
from scipy.io import loadmat
from container_models.base import ConfigBaseModel
from container_models.scan_image import ScanImage
from conversion.data_formats import (
    Mark,
    MarkStriationType,
    MarkImpressionType,
    MarkType,
)
from .utils import (
    check_if_file_exists,
    load_json,
    load_compressed_binary,
    save_as_compressed_binary,
    save_as_json,
)


def _parse_mark_type(value: Any) -> MarkType:
    if isinstance(value, (MarkImpressionType, MarkStriationType)):
        return value
    name = str(value).upper()
    if name in MarkImpressionType.__members__:
        return MarkImpressionType[name]
    if name in MarkStriationType.__members__:
        return MarkStriationType[name]
    valid = list(MarkImpressionType.__members__) + list(MarkStriationType.__members__)
    raise ValueError(f"Invalid MarkType: '{value}'. Must be one of {valid}")


class ExportedMarkData(ConfigBaseModel):
    """Validated data structure for exported Mark metadata."""

    mark_type: Annotated[MarkType, BeforeValidator(_parse_mark_type)]
    center: tuple[float, float]
    scale_x: float = Field(..., gt=0)
    scale_y: float = Field(..., gt=0)
    meta_data: dict[str, Any] = Field(default_factory=dict)


def save_mark(mark: Mark, path: Path) -> None:
    """
    Save a Mark object to JSON and NPZ files.

    Creates two files:
    - {path}.json: Contains mark metadata
    - {path}.npz: Contains compressed binary image data

    :param mark: Mark object to save
    :param path: File path (suffix is replaced for each output)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    save_as_json(data=mark.export(), file_path=path)
    save_as_compressed_binary(array=mark.scan_image.data, file_path=path)


def load_mark_from_mat_file(path: Path) -> Mark:
    """Load a `Mark` object from a .mat file."""
    parsed = loadmat(str(path))
    container = parsed["data_struct"][0, 0]
    mark = Mark(
        scan_image=ScanImage(
            data=np.asarray(container["depth_data"], dtype=np.float64),
            scale_x=float(container["xdim"][0]),
            scale_y=float(container["ydim"][0]),
        ),
        mark_type=_parse_mark_type(str(container["mark_type"][0]).lower()),
        # TODO: Parse `center` and `meta_data` from data struct
    )
    return mark


def load_mark_from_path(path: Path, stem: str) -> Mark:
    """
    Load a Mark object from JSON and NPZ files.

    Expects two files:
    - {stem}.json: Mark metadata
    - {stem}.npz: Compressed binary image data

    :param path: Directory path containing the files
    :param stem: Base filename
    :returns: Reconstructed Mark object
    :raises FileNotFoundError: If JSON or NPZ file does not exist
    """
    file_path = path / stem

    # Load and validate JSON metadata
    check_if_file_exists(json_file := file_path.with_suffix(".json"))
    meta = ExportedMarkData(**load_json(json_file))

    # Load binary image data
    check_if_file_exists(npz_file := file_path.with_suffix(".npz"))
    data = load_compressed_binary(npz_file)

    # Reconstruct Mark object
    mark = Mark(
        scan_image=ScanImage(
            data=data,
            scale_x=meta.scale_x,
            scale_y=meta.scale_y,
        ),
        mark_type=meta.mark_type,
        meta_data=meta.meta_data,
        center=meta.center,
    )

    return mark
