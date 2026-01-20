from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, Field, HttpUrl, RootModel

from models import validate_file_extension, validate_relative_path


class SupportedExtension(StrEnum):
    X3P = auto()
    PNG = auto()


type RelativePath = Annotated[
    Path,
    AfterValidator(validate_relative_path),
    AfterValidator(lambda filename: validate_file_extension(filename, SupportedExtension)),
    Field(
        ...,
        description=f"Filename of type: {', '.join(SupportedExtension)}",
        examples=["example.png", "scan.x3p"],
    ),
]


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
