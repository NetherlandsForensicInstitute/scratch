from enum import StrEnum, auto
from pathlib import Path
from typing import Self

from pydantic import DirectoryPath, FilePath, field_validator, model_validator

from models import BaseModelConfig


class SupportedExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


class UploadScan(BaseModelConfig):
    scan_file: FilePath
    output_dir: DirectoryPath

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is off a supported type."""
        extensions = tuple(f".{extension}" for extension in SupportedExtension)
        if scan_file.suffix not in extensions:
            raise ValueError(f"unsupported extension: {scan_file.name}")
        return scan_file


class ProcessScan(BaseModelConfig):
    x3p_image: FilePath
    preview_image: FilePath
    surfacemap_image: FilePath

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
