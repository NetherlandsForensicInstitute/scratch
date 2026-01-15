from container_models.scan_image import ScanImage
from .data_formats import Mark, MarkType, CropType
from pathlib import Path
import json
import numpy as np

import os
from pydantic import Field, field_validator
from pydantic import BaseModel


class ExportedScanImageData(BaseModel):
    scale_x: float
    scale_y: float


class ExportedMarkData(BaseModel):
    mark_type: MarkType
    crop_type: CropType
    center: tuple[float, float]
    scan_image: ExportedScanImageData
    meta_data: dict = Field(default_factory=dict)

    @field_validator("mark_type", mode="before")
    @classmethod
    def validate_mark_type(cls, mark_type: str) -> MarkType:
        if mark_type.lower() not in MarkType:
            raise ValueError(f"Unsupported mark type: {mark_type}")
        return MarkType[mark_type]

    @field_validator("crop_type", mode="before")
    @classmethod
    def validate_crop_type(cls, crop_type: str) -> CropType:
        if crop_type.lower() not in CropType:
            raise ValueError(f"Unsupported crop type: {crop_type}")
        return CropType[crop_type]


def save_mark(mark: Mark, path: Path, name: str):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    with open(file_path + ".json", "w") as f:
        f.write(mark.to_json())
    with open(file_path + ".npz", "wb") as f:
        np.savez_compressed(f, data=mark.scan_image.data)


def load_mark(path: Path, name: str):
    """Load a Mark object from JSON + NPZ files."""
    filepath = os.path.join(path, name)
    json_file = filepath + ".json"
    binary_file = filepath + ".npz"

    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"{json_file} not found")
    if not os.path.isfile(binary_file):
        raise FileNotFoundError(f"{binary_file} not found")

    # Load JSON metadata
    with open(json_file, "r") as f:
        meta = ExportedMarkData(**json.load(f))

    # Load array data
    with open(binary_file, "rb") as f:
        zipped = np.load(f)
        data = zipped["data"]

    return Mark(
        scan_image=ScanImage(
            data=data,
            scale_x=meta.scan_image.scale_x,
            scale_y=meta.scan_image.scale_y,
        ),
        crop_type=meta.crop_type,
        mark_type=meta.mark_type,
        _center=meta.center,
        meta_data=meta.meta_data,
    )
