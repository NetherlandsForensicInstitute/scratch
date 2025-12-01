from enum import StrEnum, auto
from pathlib import Path
from typing import Self

from pydantic import DirectoryPath, Field, FilePath, PositiveInt, field_validator, model_validator
from utils.array_definitions import ScanMap2DArray

from models import BaseModelConfig


class SupportedExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


class Level(StrEnum):
    PLAIN = auto()
    SPHERE = auto()


class Filter(StrEnum):
    RO = auto()
    R1 = auto()
    R2 = auto()


class UploadScan(BaseModelConfig):
    scan_file: FilePath = Field(
        ...,
        description="Upload scan file.",
        examples=[Path("./temp/scan.al3d"), Path("./temp/scan.x3p"), Path("./temp/scan.sur"), Path("./temp/scan.plu")],
    )
    output_dir: DirectoryPath = Field(
        ..., description="Upload output directory.", examples=[Path("./documents/project_x")]
    )

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is off a supported type."""
        if scan_file.suffix[1:] not in SupportedExtension:
            raise ValueError(f"unsupported extension: {scan_file.name}")
        return scan_file


class ProcessedDataLocation(BaseModelConfig):
    x3p_image: FilePath = Field(
        ..., description="converted subsampled X3P image.", examples=[Path("./documents/project_x/x3p.png")]
    )
    preview_image: FilePath = Field(
        ...,
        description="rgba image made from the x3p converted file.",
        examples=[Path("./documents/project_x/preview.png")],
    )
    surfacemap_image: FilePath = Field(
        ...,
        description="surface image made from the x3p converted file.",
        examples=[Path("./documents/project_x/surfacemap.png")],
    )

    @model_validator(mode="after")
    def same_parent_directory(self) -> Self:
        """Validate that all files are in the same parent directory."""
        if not all(
            getattr(self, field_name).parent == self.x3p_image.parent
            for field_name, field_info in self.__class__.model_fields.items()
            if field_info.annotation is Path
        ):
            raise ValueError("All fields must point to the same output directory")
        return self


class EditImage(BaseModelConfig):
    parsed_file: FilePath
    sampling: PositiveInt = Field(4, description="")
    level: Level | None = Field(None, description="")
    filter: Filter | None = Field(None, description="")
    zoom: bool = Field(False, description="")
    mask_array: ScanMap2DArray | None = Field(None, description="")

    @field_validator("mask_array")
    @classmethod
    def mask_within_spectrum(cls, mask_array: ScanMap2DArray | None) -> ScanMap2DArray | None:
        """Check that mask_array is 0 <=values <= 255."""
        if mask_array is None:
            return mask_array

        if ((mask_array < 0) | (mask_array > 255)).any():  # noqa
            raise ValueError("mask_array values must be between 0 and 255 (inclusive)")

        return mask_array

    @field_validator("parsed_file")
    @classmethod
    def validate_x3p_extension(cls, parse_file: FilePath) -> FilePath:
        """Validate given file is x3p extension."""
        if parse_file.suffix[1:] != SupportedExtension.X3P:
            raise ValueError(f"was expecting an x3p file: {parse_file.name}")
        return parse_file
