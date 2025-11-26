from enum import StrEnum, auto
from pathlib import Path
from typing import Self

from pydantic import DirectoryPath, Field, FilePath, field_validator, model_validator

from models import BaseModelConfig


class SupportedExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


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


class ProcessScan(BaseModelConfig):
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

    @property
    def output_directory(self) -> Path:
        return self.x3p_image.parent

    @model_validator(mode="after")
    def same_parent_directory(self) -> Self:
        """Validate that all files are in the same parent directory."""
        if not all(
            getattr(self, field_name).parent == self.output_directory
            for field_name, field_info in self.__class__.model_fields.items()
            if field_info.annotation is Path
        ):
            raise ValueError("All fields must point to the same output directory")
        return self
