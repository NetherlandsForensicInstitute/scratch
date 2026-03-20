from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, Field, HttpUrl, SerializerFunctionWrapHandler, model_serializer

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
        description="RGBA preview image rendered from the parsed scan surface data.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/preview.png"],
    )
    surface_map_image: HttpUrl = Field(
        ...,
        description="Height-map visualization of the scan surface.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/surface_map.png"],
    )


class ProcessedDataAccess(GeneratedImages):
    scan_image: HttpUrl = Field(
        ...,
        description="Subsampled X3P scan file, converted from the original upload.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/scan.x3p"],
    )


class PrepareMarkResponse(GeneratedImages):
    """Response model for prepared mark data access."""

    mark_data: HttpUrl = Field(
        ...,
        description="Cropped, rotated, and resampled mark data before surface filtering.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark.npz"],
    )
    mark_meta: HttpUrl = Field(
        ...,
        description="Metadata for the mark data.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark.json"],
    )
    processed_data: HttpUrl = Field(
        ...,
        description="Mark surface data after filtering and preprocessing.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/processed.npz"],
    )
    processed_meta: HttpUrl = Field(
        ...,
        description="Metadata for the processed mark data.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/processed.json"],
    )


class PrepareMarkResponseStriation(PrepareMarkResponse):
    """Response model for prepared striation mark data access."""

    profile_data: HttpUrl = Field(
        ...,
        description="Mean or median profile of a striation mark.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/profile.npz"],
    )


class PrepareMarkResponseImpression(PrepareMarkResponse):
    """Response model for prepared impression mark data access."""

    leveled_data: HttpUrl = Field(
        ...,
        description="Leveled impression mark surface (before surface filtering is applied).",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/leveled.npz"],
    )
    leveled_meta: HttpUrl = Field(
        ...,
        description="Metadata for the leveled impression mark data.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/leveled.json"],
    )


class ComparisonResponse(URLContainer):
    """Response model for comparison data access."""

    filtered_reference_heatmap: HttpUrl = Field(
        ...,
        description="Heatmap of the reference mark after surface filtering.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/filtered_reference_heatmap.png"
        ],
    )
    comparison_overview: HttpUrl = Field(
        ...,
        description="Combined overview figure showing all comparison results.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/comparison_overview.png"
        ],
    )


class ComparisonResponseImpressionURL(ComparisonResponse):
    raw_reference_heatmap: HttpUrl = Field(
        ...,
        description="Heatmap of the raw reference mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/raw_reference_heatmap.png"
        ],
    )
    raw_compared_heatmap: HttpUrl = Field(
        ...,
        description="Heatmap of the raw compared mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/raw_compared_heatmap.png"
        ],
    )
    filtered_compared_heatmap: HttpUrl = Field(
        ...,
        description="Heatmap of the compared mark after surface filtering.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/filtered_compared_heatmap.png"
        ],
    )
    cell_reference_heatmap: HttpUrl = Field(
        ...,
        description="Heatmap of the reference mark after cell-level preprocessing.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/cell_reference_heatmap.png"
        ],
    )
    cell_compared_heatmap: HttpUrl = Field(
        ...,
        description="Heatmap of the compared mark after cell-level preprocessing.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/cell_compared_heatmap.png"
        ],
    )
    cell_overlay: HttpUrl = Field(
        ...,
        description="Surface overlay showing the cell grid with CMC classification status.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/cell_overlay.png"],
    )
    cell_cross_correlation: HttpUrl = Field(
        ...,
        description="Cell-based cross-correlation heatmap.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/cell_cross_correlation.png"
        ],
    )


class ComparisonResponseImpression(URLContainer):
    urls: ComparisonResponseImpressionURL
    cells: list[dict] = Field(
        default_factory=list,
        description="Per-cell CMC results for use in LR calculation.",
    )
    comparison_results: dict = Field(
        default_factory=dict,
        description="Impression comparison metrics including CMC counts, fractions, and consensus registration.",
    )

    @model_serializer(mode="wrap")
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Serialize model to flat json."""
        data = handler(self)
        return {
            **data["urls"],
            "cells": data["cells"],
            "comparison_results": data["comparison_results"],
        }


class ComparisonResponseStriationURL(ComparisonResponse):
    mark_reference_aligned_surfacemap: HttpUrl = Field(
        ...,
        description="Surface map data of the reference mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_reference_aligned_surfacemap.json"
        ],
    )
    mark_compared_aligned_surfacemap: HttpUrl = Field(
        ...,
        description="Surface map data of the compared mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_compared_aligned_surfacemap.json"
        ],
    )
    mark_reference_aligned_data: HttpUrl = Field(
        ...,
        description="Aligned reference mark surface data.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_reference_aligned.npz"
        ],
    )
    mark_reference_aligned_meta: HttpUrl = Field(
        ...,
        description="Metadata for the aligned reference mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_reference_aligned.json"
        ],
    )
    mark_compared_aligned_data: HttpUrl = Field(
        ...,
        description="Aligned compared mark surface data.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_compared_aligned.npz"
        ],
    )
    mark_compared_aligned_meta: HttpUrl = Field(
        ...,
        description="Metadata for the aligned compared mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_compared_aligned.json"
        ],
    )
    mark_reference_aligned_preview: HttpUrl = Field(
        ...,
        description="Preview image of the reference mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_reference_aligned_preview.png"
        ],
    )
    mark_compared_aligned_preview: HttpUrl = Field(
        ...,
        description="Preview image of the compared mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_compared_aligned_preview.png"
        ],
    )
    similarity_plot: HttpUrl = Field(
        ...,
        description="Plot of aligned striation profiles overlaid for visual comparison.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/similarity_plot.png"],
    )
    filtered_compared_heatmap: HttpUrl = Field(
        ...,
        description="Heatmap of the compared mark after surface filtering.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/filtered_compared_heatmap.png"
        ],
    )
    side_by_side_heatmap: HttpUrl = Field(
        ...,
        description="Side-by-side heatmap of both marks for visual comparison.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/side_by_side_heatmap.png"
        ],
    )


class ComparisonResponseStriation(URLContainer):
    urls: ComparisonResponseStriationURL
    comparison_results: dict = Field(
        default_factory=dict,
        description="Striation comparison metrics including correlation, roughness, and alignment geometry.",
    )

    @model_serializer(mode="wrap")
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Serialize model to flat json."""
        data = handler(self)
        return {
            **data["urls"],
            "comparison_results": data["comparison_results"],
        }


class LRResponseURL(URLContainer):
    lr_overview_plot: HttpUrl = Field(
        ...,
        description="Overview plot showing the log-likelihood ratio against the reference population distributions.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/lr_overview_plot.png"],
    )


class LRResponse(BaseModel):
    urls: LRResponseURL
    llr: float = Field(
        ...,
        description="Log10 likelihood ratio.",
    )
    llr_lower_ci: float | None = Field(
        None,
        description="Lower bound of the log10 likelihood ratio confidence interval, or null if not computed.",
    )
    llr_upper_ci: float | None = Field(
        None,
        description="Upper bound of the log10 likelihood ratio confidence interval, or null if not computed.",
    )

    @model_serializer(mode="wrap")
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Serialize model to flat json."""
        data = handler(self)
        urls = data.pop("urls")
        return {**urls, **data}
