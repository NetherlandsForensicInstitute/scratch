from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, TypeVar

from pydantic import AfterValidator, BaseModel, Field, HttpUrl, model_serializer

from models import (
    validate_file_extension,
    validate_relative_path,
)


class SupportedExtension(StrEnum):
    X3P = auto()
    PNG = auto()
    JSON = auto()
    NPZ = auto()


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

C = TypeVar("C", bound="URLContainer")


class URLContainer(BaseModel):
    @classmethod
    def from_enum(
        cls: type[C],
        enum: type[StrEnum],
        base_url: str,
    ) -> C:
        """Initiate the Response model with the given files from the enum."""
        return cls(**{file.name: HttpUrl(f"{base_url}/{file.value}") for file in enum})


class GeneratedImages(URLContainer):
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


class ProcessedDataAccess(GeneratedImages):
    scan_image: HttpUrl = Field(
        ...,
        description="converted subsampled X3P image.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/scan.x3p"],
    )


class PrepareMarkResponse(GeneratedImages):
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


class ComparisonResponse(URLContainer):
    """Response model for comparison data access."""

    filtered_reference_heatmap: HttpUrl = Field(
        description="",
        examples=[
            "http://localhost:8000/preprocessor/files/surface_comparator_859lquto/filtered_reference_heatmap.json"
        ],
    )
    comparison_overview: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/comparison_overview.png"],
    )


class ComparisonResponseImpressionURL(ComparisonResponse):
    leveled_reference_heatmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/leveled_reference_heatmap.png"],
    )
    leveled_compared_heatmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/leveled_compared_heatmap.png"],
    )
    filtered_compared_heatmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/filtered_compared_heatmap.png"],
    )
    cell_reference_heatmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/cell_reference_heatmap.png"],
    )
    cell_compared_heatmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/cell_compared_heatmap.png"],
    )
    cell_overlay: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/cell_overlay.png"],
    )
    cell_cross_correlation: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/cell_cross_correlation.png"],
    )


class ComparisonResponseImpression(URLContainer):
    urls: ComparisonResponseImpressionURL
    cells: list[dict] = Field(
        default_factory=list,
        description="Per-cell CMC results for use in LR calculation.",
    )

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        """Serialize model to flat json."""
        data = handler(self)
        return {
            **data["urls"],
            "cells": data["cells"],
        }


class ComparisonResponseStriationURL(ComparisonResponse):
    mark_ref_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark_ref_surfacemap.json"],
    )
    mark_comp_surfacemap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark_comp_surfacemap.json"],
    )
    mark_reference_aligned_data: HttpUrl = Field(
        ...,
        description="Aligned reference mark.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark_reference_aligned.npz"],
    )
    mark_reference_aligned_meta: HttpUrl = Field(
        ...,
        description="meta data from the aligned reference mark data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark_reference_aligned.json"],
    )
    mark_compared_aligned_data: HttpUrl = Field(
        ...,
        description="Aligned compared mark.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark_compared_aligned.npz"],
    )
    mark_compared_aligned_meta: HttpUrl = Field(
        ...,
        description="meta data from the aligned compared mark data.",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark_compared_aligned.json"],
    )
    mark_ref_preview: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark_ref_preview.png"],
    )
    mark_comp_preview: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/mark_comp_preview.png"],
    )
    similarity_plot: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/similarity_plot.png"],
    )
    filtered_compared_heatmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/filtered_compared_heatmap.png"],
    )
    side_by_side_heatmap: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/side_by_side_heatmap.png"],
    )


class ComparisonResponseStriation(URLContainer):
    urls: ComparisonResponseStriationURL
    comparison_results: dict = Field(
        default_factory=dict,
        description="Striation comparison metrics including correlation, roughness, and alignment geometry.",
    )

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        """Serialize model to flat json."""
        data = handler(self)
        return {
            **data["urls"],
            "comparison_results": data["comparison_results"],
        }


class LRResponseURL(URLContainer):
    lr_overview_plot: HttpUrl = Field(
        description="",
        examples=["http://localhost:8000/preprocessor/files/surface_comparator_859lquto/lr_overview_plot.png"],
    )


class LRResponse(BaseModel):
    urls: LRResponseURL
    lr: float

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        """Serialize model to flat json."""
        data = handler(self)
        return {
            **data["urls"],
            "lr": data["lr"],
        }
