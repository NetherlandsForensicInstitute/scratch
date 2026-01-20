"""
Module for serializing and deserializing Mark objects to/from disk.

This module provides functionality to save Mark objects as JSON metadata
with NPZ binary data, and load them back into memory.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from container_models.base import ConfigBaseModel
from container_models.scan_image import ScanImage
from conversion.data_formats import CropType, Mark, MarkType
from .utils import (
    check_if_file_exists,
    load_json,
    load_compressed_binary,
    save_as_compressed_binary,
    save_as_json,
)


class ExportedMarkData(ConfigBaseModel):
    """Validated data structure for exported Mark metadata."""

    mark_type: MarkType
    crop_type: CropType
    center: tuple[float, float]
    scale_x: float = Field(..., gt=0)
    scale_y: float = Field(..., gt=0)
    meta_data: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def _validate_enum_value(value: str, enum_class: type[Enum]) -> None:
        """Validate that a string value exists in an enum."""
        if value.upper() not in enum_class.__members__:
            raise ValueError(
                f"Invalid {enum_class.__name__}: '{value}'. "
                f"Must be one of {list(enum_class.__members__.keys())}"
            )

    @field_validator("mark_type", mode="before")
    @classmethod
    def validate_mark_type(cls, value: str) -> MarkType:
        """Validate and convert mark_type to MarkType enum."""
        cls._validate_enum_value(value, MarkType)
        return MarkType[value.upper()]

    @field_validator("crop_type", mode="before")
    @classmethod
    def validate_crop_type(cls, value: str) -> CropType:
        """Validate and convert crop_type to CropType enum."""
        cls._validate_enum_value(value, CropType)
        return CropType[value.upper()]


def save_mark(mark: Mark, path: Path, stem: str) -> None:
    """
    Save a Mark object to JSON and NPZ files.

    Creates two files:
    - {stem}.json: Contains mark metadata
    - {stem}.npz: Contains compressed binary image data

    :param mark: Mark object to save
    :param path: Directory path where files will be saved
    :param stem: Base filename
    """
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / stem

    save_as_json(data=mark.export(), file_path=file_path)
    save_as_compressed_binary(array=mark.scan_image.data, file_path=file_path)


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
        crop_type=meta.crop_type,
        mark_type=meta.mark_type,
        meta_data=meta.meta_data,
    )
    mark._center = meta.center

    return mark
