from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Self, cast

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


class BaseResponseURLs(BaseModelConfig):
    @classmethod
    def get_files(cls, resource_directory: Path) -> dict[str, Path]:
        """
        Generate file paths within this directory's resource path.

        Creates a mapping of field names to absolute file paths by joining
        each filename of the base model with this directory's resource path.

        :param resource_directory: The resource directory path.
        :return: Dictionary mapping field names to absolute Path objects.
        """
        return {
            field.alias or name: resource_directory / cast(dict, field.json_schema_extra)["file_name"]
            for name, field in cls.model_fields.items()
        }

    @classmethod
    def generate_urls(cls, access_url: str, **kwargs) -> Self:
        """
        Generate access URLs for files within this directory.

        Creates a mapping of field names to HTTP URLs by joining each filename of the base model
        with this directory's access URL base path.

        :param access_url: The base access URL for the directory.
        :return: Dictionary mapping field names to validated HttpUrl objects.
        .. note::
        URLs are validated as proper HTTP URLs via Pydantic's HttpUrl type.
        """
        return cls(
            **{
                field.alias or name: HttpUrl(url=f"{access_url}/{cast(dict, field.json_schema_extra)['file_name']}")
                for name, field in cls.model_fields.items()
                if isinstance(field.json_schema_extra, dict)
            }
            | kwargs
        )


class GeneratedImages(BaseResponseURLs):
    preview_image: HttpUrl = Field(
        ...,
        description="rgba image render from the parsed scan data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/preview.png"],
        alias="preview",
        json_schema_extra={"file_name": "preview.png"},
    )
    surface_map_image: HttpUrl = Field(
        ...,
        description="surface image render from the scan data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/surface_map.png"],
        alias="surface_map",
        json_schema_extra={"file_name": "surface_map.png"},
    )


class ProcessedDataAccess(GeneratedImages):
    scan_image: HttpUrl = Field(
        ...,
        description="converted subsampled X3P image.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/scan.x3p"],
        alias="scan",
        json_schema_extra={"file_name": "scan.x3p"},
    )


class PrepareMarkResponse(GeneratedImages):
    """Response model for prepared mark data access."""

    mark_data: HttpUrl = Field(
        ...,
        description="Mark without preprocessing, only cropped, rotated and resampled.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark.npz"],
        json_schema_extra={"file_name": "mark.npz"},
    )
    mark_meta: HttpUrl = Field(
        ...,
        description="meta data from the mark data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark.json"],
        json_schema_extra={"file_name": "mark.json"},
    )
    processed_data: HttpUrl = Field(
        ...,
        description="Preprocessed mark (impression or striation) after filtering and processing.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/processed.npz"],
        json_schema_extra={"file_name": "processed.npz"},
    )
    processed_meta: HttpUrl = Field(
        ...,
        description="meta data from the processed mark data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/processed.json"],
        json_schema_extra={"file_name": "processed.json"},
    )


class PrepareMarkResponseStriation(PrepareMarkResponse):
    """Response model for prepared striation mark data access."""

    profile_data: HttpUrl = Field(
        ...,
        description="Mean or median profile of a striation mark.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.npz"],
        json_schema_extra={"file_name": "profile.npz"},
    )


class PrepareMarkResponseImpression(PrepareMarkResponse):
    """Response model for prepared impression mark data access."""

    leveled_data: HttpUrl = Field(
        ...,
        description="Leveled impression mark (same as processed but without filtering).",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/leveled.npz"],
        json_schema_extra={"file_name": "leveled.npz"},
    )
    leveled_meta: HttpUrl = Field(
        ...,
        description="meta data from the leveled impression mark data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/leveled.json"],
        json_schema_extra={"file_name": "leveled.json"},
    )


class ComparisonResponse(BaseResponseURLs):
    """Response model for comparison data access."""

    mark_ref_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.json"],
        json_schema_extra={"file_name": "mark_ref_surfacemap.png"},
    )
    mark_comp_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.json"],
        json_schema_extra={"file_name": "mark_comp_surfacemap.png"},
    )
    mark_ref_filtered_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.json"],
        json_schema_extra={"file_name": "mark_ref_filtered_surfacemap.png"},
    )
    comparison_overview: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "comparison_overview.png"},
    )


class ComparisonResponseImpression(ComparisonResponse):
    mark_ref_filtered_moved_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark_ref_filtered_moved_surfacemap.png"},
    )
    mark_ref_filtered_bb_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark_ref_filtered_bb_surfacemap.png"},
    )
    mark_comp_filtered_bb_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark_comp_filtered_bb_surfacemap.png"},
    )
    mark_comp_filtered_all_bb_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark_comp_filtered_all_bb_surfacemap.png"},
    )
    cell_accf_distribution: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "cell_accf_distribution.png"},
    )


class ComparisonResponseStriation(ComparisonResponse):
    mark_ref_depthmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark_ref_depthmap.png"},
    )
    mark_comp_depthmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark_ref_depthmap.png"},
    )
    similarity_plot: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "similarity_plot.png"},
    )
    mark_comp_filtered_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark_comp_filtered_surfacemap.png"},
    )
    mark1_vs_moved_mark2: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark1_vs_moved_mark2.png"},
    )
    wavedlength_plot: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "mark1_vs_moved_mark2.png"},
    )


class LRResponse(BaseResponseURLs):
    lr_overview_plot: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/profile.png"],
        json_schema_extra={"file_name": "lr_overview_plot.png"},
    )
    lr: float
