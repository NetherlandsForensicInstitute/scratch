from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, NamedTuple, Protocol

from pydantic import AfterValidator, Field, HttpUrl

from models import BaseModelConfig, validate_file_extension


class SupportedExtension(StrEnum):
    X3P = auto()
    PNG = auto()


type FileName = Annotated[
    str,
    AfterValidator(lambda filename: validate_file_extension(filename, SupportedExtension)),
    Field(
        ...,
        description=f"Filename of type: {', '.join(SupportedExtension)}",
        examples=["example.png", "scan.x3p"],
    ),
]


class ImageAccess(NamedTuple):
    resource_path: Path
    access_url: str


class ProcessData(Protocol):
    @property
    def surface_map_filename(self) -> str: ...

    @property
    def preview_filename(self) -> str: ...

    @property
    def x3p_filename(self) -> str: ...


class ProcessedDataAccess(BaseModelConfig):
    x3p_image: HttpUrl = Field(
        ...,
        description="converted subsampled X3P image.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/scan.x3p"],
    )
    preview_image: HttpUrl = Field(
        ...,
        description="rgba image render from the parsed scan data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/preview.png"],
    )
    surface_map_image: HttpUrl = Field(
        ...,
        description="surface image render from the scan data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/surface_map.png"],
    )

    @classmethod
    def from_access_point(cls, access_url: str, process_data: ProcessData) -> ProcessedDataAccess:
        """
        Create ProcessedDataAccess from access URL and process data.

        Constructs a ProcessedDataAccess instance with complete URLs for all processed files
        by combining the base access URL with filenames from the process data.

        :param access_url: Base URL for file retrieval endpoints.
        :param process_data: ProcessData protocol containing file naming information.
        :returns: ProcessedDataAccess with complete URLs for x3p, preview, and surface_map files.
        """
        return ProcessedDataAccess(
            x3p_image=HttpUrl(f"{access_url}/{process_data.x3p_filename}"),
            preview_image=HttpUrl(f"{access_url}/{process_data.preview_filename}"),
            surface_map_image=HttpUrl(f"{access_url}/{process_data.surface_map_filename}"),
        )
