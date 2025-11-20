from enum import StrEnum, auto

from pydantic import DirectoryPath, FilePath, field_validator

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
