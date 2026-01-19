"""Module for serializing and deserializing Mark objects to/from disk.

This module provides functionality to save Mark objects as JSON metadata
with NPZ binary data, and load them back into memory.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from container_models.base import ScanMap2DArray
from container_models.scan_image import ScanImage

from conversion.data_formats import CropType, Mark, MarkType


class ScanImageMetaData(BaseModel):
    """Metadata for scan image scaling information."""

    scale_x: float
    scale_y: float


class ExportedMarkData(BaseModel):
    """Validated data structure for exported Mark metadata."""

    mark_type: MarkType
    crop_type: CropType
    center: tuple[float, float]
    scan_image: ScanImageMetaData
    meta_data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mark_type", mode="before")
    @classmethod
    def validate_mark_type(cls, mark_type: str | MarkType) -> MarkType:
        """
        Validate and convert `mark_type` to MarkType enum.

        :param mark_type: String or MarkType enum value
        :returns: MarkType enum value
        :raises ValueError: If mark_type is not a valid MarkType
        """
        if isinstance(mark_type, MarkType):
            return mark_type
        if mark_type.upper() not in MarkType.__members__:
            raise ValueError(f"Unsupported mark type: {mark_type}")
        return MarkType[mark_type.upper()]

    @field_validator("crop_type", mode="before")
    @classmethod
    def validate_crop_type(cls, crop_type: str | CropType) -> CropType:
        """
        Validate and convert `crop_type` to CropType enum.

        :param crop_type: String or CropType enum value
        :returns: CropType enum value
        :raises ValueError: If crop_type is not a valid CropType
        """
        if isinstance(crop_type, CropType):
            return crop_type
        if crop_type.upper() not in CropType.__members__:
            raise ValueError(f"Unsupported crop type: {crop_type}")
        return CropType[crop_type.upper()]


def save(mark: Mark, path: Path, name: str) -> None:
    """Save a Mark object to JSON and NPZ files.

    :param mark: Mark object to save
    :param path: Directory path where files will be saved
    :param name: Base filename (without extension)
    """
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / name
    _save_json(mark, file_path)
    _save_binary(mark, file_path)


def from_path(path: Path, name: str) -> Mark:
    """
    Load a Mark object from JSON and NPZ files.

    :param path: Directory path containing the files
    :param name: Base filename (without extension)
    :returns: Reconstructed Mark object
    :raises FileNotFoundError: If JSON or NPZ file does not exist
    """
    file_path = path / name

    # Load and validate metadata
    json_file = file_path.with_suffix(".json")
    _check_file_exists(json_file)
    meta = _load_json(json_file)

    # Load binary data
    npz_file = file_path.with_suffix(".npz")
    _check_file_exists(npz_file)
    data = _load_binary(npz_file)

    mark = Mark(
        scan_image=ScanImage(
            data=data,
            scale_x=meta.scan_image.scale_x,
            scale_y=meta.scan_image.scale_y,
        ),
        crop_type=meta.crop_type,
        mark_type=meta.mark_type,
        meta_data=meta.meta_data,
    )
    mark._center = meta.center
    return mark


def _check_file_exists(file_path: Path) -> None:
    """
    Check if a file exists, raise FileNotFoundError if not.

    :param file_path: Path to check
    :raises FileNotFoundError: If file does not exist
    """
    if not file_path.is_file():
        raise FileNotFoundError(f'File "{file_path}" does not exist')


def _to_json(mark: Mark) -> str:
    """
    Convert Mark object to JSON string.

    :param mark: Mark object to convert
    :returns: JSON string representation
    """
    data = {
        "mark_type": mark.mark_type.name,
        "crop_type": mark.crop_type.name,
        "center": mark.center,
        "scan_image": {
            "scale_x": mark.scan_image.scale_x,
            "scale_y": mark.scan_image.scale_y,
        },
        "meta_data": mark.meta_data,
    }
    return json.dumps(data, indent=4)


def _load_json(file_path: Path) -> ExportedMarkData:
    """
    Load and validate JSON metadata.

    :param file_path: Path to JSON file
    :returns: Validated ExportedMarkData object
    """
    with file_path.open("r") as f:
        return ExportedMarkData(**json.load(f))


def _load_binary(file_path: Path) -> ScanMap2DArray:
    """
    Load numpy array from NPZ file.

    :param file_path: Path to NPZ file
    :returns: Numpy array containing scan image data
    """
    with np.load(file_path) as zipped:
        return zipped["data"]


def _save_json(mark: Mark, file_path: Path) -> None:
    """
    Save Mark metadata as JSON file.

    :param mark: Mark object to save
    :param file_path: Base file path (extension will be added)
    """
    json_path = file_path.with_suffix(".json")
    json_path.write_text(_to_json(mark))


def _save_binary(mark: Mark, file_path: Path) -> None:
    """
    Save Mark scan image data as compressed NPZ file.

    :param mark: Mark object to save
    :param file_path: Base file path (extension will be added)
    """
    npz_path = file_path.with_suffix(".npz")
    np.savez_compressed(npz_path, data=mark.scan_image.data)
