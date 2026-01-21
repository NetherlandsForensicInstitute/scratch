from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from pydantic import (
    UUID4,
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    HttpUrl,
    RootModel,
    StringConstraints,
)

from constants import EXTRACTOR_ROUTE
from settings import get_settings


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        regex_engine="rust-regex",
        extra="forbid",
    )


class SupportedScanExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


def validate_file_extension(filename: Path, extensions: type[StrEnum]) -> Path:
    """
    Validate that the file has a supported extension.

    :param filename: The filename to validate.
    :param extensions: Supported file extensions enum.
    :return: The validated filename.
    :raises ValueError: If the filename does not have a supported extension.
    """
    if filename.suffix not in (f".{ext}" for ext in extensions):
        raise ValueError(f"unsupported file type: {filename}, try: {', '.join(extensions)}")
    return filename


def validate_not_executable(filepath: Path) -> Path:
    """
    Validate that the file is not an executable.

    Checks for executable magic bytes including shebang, ELF (Linux),
    PE (Windows), and Mach-O (macOS) binaries.

    :param filepath: The file to validate.
    :return: The validated filepath.
    :raises ValueError: If the file contains executable magic bytes.
    """
    with filepath.open("rb") as f:
        header = f.read(4)

    # Check for executable magic bytes: shebang, ELF (Linux), PE (Windows), Mach-O (macOS)
    executable_pattern = re.compile(
        b"^#!"  # Shebang
        b"|^\x7fELF"  # ELF
        b"|^MZ"  # PE
        b"|^(\xfe\xed\xfa\xce|\xfe\xed\xfa\xcf|\xce\xfa\xed\xfe|\xcf\xfa\xed\xfe)"  # Mach-O
    )
    if executable_pattern.match(header):
        raise ValueError(f"executable files are not allowed: {filepath.name}")
    return filepath


def validate_relative_path(filepath: Path) -> Path:
    """
    Validate that the path is relative, not absolute.

    :param filepath: The path to validate.
    :return: The validated relative path.
    :raises ValueError: If the path is absolute.
    """
    if filepath.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {filepath}")
    return filepath


type ProjectTag = Annotated[str, StringConstraints(pattern=r"")]
type ScanFile = Annotated[
    FilePath,
    AfterValidator(lambda filepath: validate_file_extension(filepath, SupportedScanExtension)),
    AfterValidator(validate_not_executable),
    Field(
        ...,
        description=f"Path to the input scan file. Supported formats: {', '.join(SupportedScanExtension)}",
    ),
]


def _generate_unique_token() -> UUID4:
    """Generate a unique token that doesn't have an existing directory."""
    storage = get_settings().storage
    while True:
        if not tuple(storage.glob((token := uuid4()).hex)):
            return token


class DirectoryAccess(BaseModelConfig):
    """
    Directory access model for managing temporary file storage.

    Provides unique token-based directory paths for file storage operations.
    Uses application settings for storage location configuration.
    """

    token: UUID4 = Field(
        default_factory=_generate_unique_token,
        description=(
            "Unique UUID4 identifier for the storage directory. "
            "Auto-generated to ensure no collisions with existing directories."
        ),
    )
    tag: ProjectTag = Field(
        ...,
        description=(
            "Project tag for directory organization. "
            "Combined with token to create unique directory names in format '{tag}-{token.hex}'."
        ),
    )

    @property
    def access_url(self) -> str:
        """Get the URL to access files in this directory."""
        return f"{get_settings().base_url}{EXTRACTOR_ROUTE}/files/{self.token}"

    @property
    def resource_path(self) -> Path:
        """Get the filesystem path for this directory."""
        return get_settings().storage / f"{self.tag}-{self.token.hex}"

    @property
    def last_modified(self) -> datetime:
        """Get the last modification time of the directory."""
        return datetime.fromtimestamp(self.resource_path.stat().st_mtime)

    def __str__(self) -> str:
        """Return string representation of the resource path."""
        return str(self.resource_path)


class ProcessDataUrls(RootModel):
    """
    Collection of HTTP URLs pointing to processed scan data files.

    Provides immutable tuple of URLs for accessing generated outputs
    such as X3P files, surface maps, and preview images.
    """

    root: tuple[HttpUrl, ...] = Field(
        ...,
        description="Tuple of HTTP URLs for accessing processed scan files (X3P, surface maps, preview images).",
        examples=[
            (
                "http://localhost:8000/extractor/files/a1b2c3d4/project/scan.x3p",
                "http://localhost:8000/extractor/files/a1b2c3d4/project/surface_map.png",
                "http://localhost:8000/extractor/files/a1b2c3d4/project/preview.png",
            ),
        ],
        min_length=1,
    )
    model_config = ConfigDict(frozen=True, regex_engine="rust-regex")
