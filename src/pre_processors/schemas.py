from enum import StrEnum, auto
from typing import Literal

from numpy import float64
from numpydantic import NDArray
from pydantic import DirectoryPath, FilePath, field_validator

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


class EditImage(BaseModelConfig):
    parsed_file: FilePath
    sampling: int = 4
    level: Level | None = None
    filter: Filter | None = None
    zoom: bool = False
    marks: NDArray[Literal["* x, * y"], float64] | None = None
