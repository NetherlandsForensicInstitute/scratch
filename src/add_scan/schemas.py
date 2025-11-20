from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, DirectoryPath, FilePath, field_validator


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


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        regex_engine="rust-regex",
        extra="forbid",
    )


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
