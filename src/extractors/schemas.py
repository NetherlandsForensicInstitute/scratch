from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, Field, HttpUrl

from models import (
    BaseModelConfig,
    validate_file_extension,
    validate_relative_path,
)


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


class ProcessedDataAccess(BaseModelConfig):
    scan_image: HttpUrl = Field(
        ...,
        description="converted subsampled X3P image.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/scan.x3p"],
        alias="scan",
    )
    preview_image: HttpUrl = Field(
        ...,
        description="rgba image render from the parsed scan data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/preview.png"],
        alias="preview",
    )
    surface_map_image: HttpUrl = Field(
        ...,
        description="surface image render from the scan data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/surface_map.png"],
        alias="surface_map",
    )


class PrepareMarkResponse(ProcessedDataAccess):
    """Response model for prepared mark data access."""

    mark_data: HttpUrl = Field(
        ...,
        description="Mark without preprocessing, only cropped, rotated and resampled.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark.npz"],
    )
    mark_meta: HttpUrl = Field(
        ...,
        description="meta data from the mark data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark.json"],
    )
    processed_data: HttpUrl = Field(
        ...,
        description="Preprocessed mark (impression or striation) after filtering and processing.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/processed.npz"],
    )
    processed_meta: HttpUrl = Field(
        ...,
        description="meta data from the processed mark data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/processed.json"],
    )


class PrepareMarkResponseStriation(PrepareMarkResponse):
    """Response model for prepared striation mark data access."""

    profile_data: HttpUrl = Field(
        ...,
        description="Mean or median profile of a striation mark.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.npz"],
    )
    profile_meta: HttpUrl = Field(
        ...,
        description="meta data from the profile data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.json"],
    )


class PrepareMarkResponseImpression(PrepareMarkResponse):
    """Response model for prepared impression mark data access."""

    leveled_data: HttpUrl = Field(
        ...,
        description="Leveled impression mark (same as processed but without filtering).",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/leveled.npz"],
    )
    leveled_meta: HttpUrl = Field(
        ...,
        description="meta data from the leveled impression mark data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/leveled.json"],
    )
