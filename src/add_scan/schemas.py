from enum import StrEnum, auto

from pydantic import BaseModel, DirectoryPath, FilePath, field_validator


class SupportedExtension(StrEnum):
    MAT = auto()
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    LMS = auto()
    PLU = auto()
    PNG = auto()
    BMP = auto()
    JPG = auto()
    JPEG = auto()


class BaseModelConfig(BaseModel): ...


class UploadScan(BaseModelConfig):
    scan_file: FilePath
    output_dir: DirectoryPath

    @field_validator("scan_file", mode="after")
    def validate_file_extension(self, scan_file: FilePath) -> FilePath:
        """Validate given file is off a supported type."""
        extensions = tuple(extension for extension in SupportedExtension)
        if scan_file.suffix not in extensions:
            raise ValueError("unsupported extension: {scan_file.name}")
        return scan_file
