from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import Annotated, Self

from conversion.data_formats import MarkMetadata
from conversion.profile_correlator import StriationComparisonResults
from conversion.surface_comparison.models import Cell, ComparisonParams
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    HttpUrl,
    NonNegativeInt,
    computed_field,
    model_validator,
)

from models import BaseModelConfig
from schemas import URLContainer


class MarkDirectories(BaseModelConfig):
    mark_dir_ref: DirectoryPath = Field(
        ...,
        description="Path to the directory containing the preprocessed reference mark files.",
    )
    mark_dir_comp: DirectoryPath = Field(
        ...,
        description="Path to the directory containing the preprocessed compared mark files.",
    )

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return "SomethingWithNoValue"


class MetadataParameters(BaseModelConfig):
    metadata_reference: MarkMetadata = Field(..., description="Metadata identifying the reference mark.")
    metadata_compared: MarkMetadata = Field(..., description="Metadata identifying the compared mark.")


class CalculateScore(MarkDirectories, MetadataParameters): ...


class CalculateScoreImpression(CalculateScore):
    comparison_params: ComparisonParams = Field(
        ...,
        description="Parameters controlling the CMC comparison (cell size, angle tolerance, etc.).",
    )


class CalculateLR(MarkDirectories, MetadataParameters):
    lr_system_path: DirectoryPath = Field(
        ...,
        description="Path to the likelihood ratio system directory containing the trained LR model.",
    )
    user_id: str = Field(
        ...,
        description="Identifier of the user performing the LR calculation.",
    )
    date_report: datetime.date = Field(
        ...,
        description="Date of the report for which the LR is calculated.",
    )


class CalculateLRImpression(CalculateLR):
    score: NonNegativeInt = Field(
        ...,
        description="CMC score (number of congruent matching cells).",
    )
    cells: Sequence[Cell] = Field(
        ...,
        description="Per-cell CMC results from the impression comparison.",
    )

    @property
    def n_cells(self) -> int:
        """Total number of cells in the comparison grid."""
        return len(self.cells)

    @model_validator(mode="after")
    def score_cannot_exceed_n_cells(self) -> Self:
        """Ensure that the score <= n_cells."""
        if self.score > self.n_cells:
            raise ValueError(f"score ({self.score}) cannot exceed n_cells ({self.n_cells})")
        return self


class CalculateLRStriation(CalculateLR):
    mark_dir_ref_aligned: DirectoryPath = Field(
        ...,
        description="Path to the directory containing the aligned reference mark files.",
    )
    mark_dir_comp_aligned: DirectoryPath = Field(
        ...,
        description="Path to the directory containing the aligned compared mark files.",
    )
    score: Annotated[
        float,
        Field(
            ge=-1,
            le=1,
            description="Correlation coefficient from the striation comparison (range: -1 to 1).",
        ),
    ]

    @property
    def tag(self) -> str:
        # TODO: incorporate mark_dir_ref_aligned and mark_dir_comp_aligned once
        # the base tag implementation is finalised.
        return super().tag


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


class ComparisonImpressionMetrics(BaseModelConfig):
    score: int = Field(
        ...,
        ge=0,
        description="CMC score: number of cells classified as congruent matching cells.",
        examples=[30],
    )
    cmc_area_fraction: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of the total valid surface area covered by congruent matching cells, range 0-1.",
        examples=[0.75],
    )
    estimated_rotation: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Estimated rotation between reference and compared mark from cell registration, in degrees.",
        examples=[1.0],
    )
    estimated_translation: tuple[float, float] = Field(
        ...,
        description="Estimated (x, y) translation between reference and compared mark from cell registration, in "
        "meters.",
        examples=[(-9.4e-6, 10.1e-6)],
    )


class ComparisonResponseImpression(URLContainer):
    urls: ComparisonResponseImpressionURL
    cells: list[Cell] = Field(
        default_factory=list,
        description="Per-cell CMC results for use in LR calculation.",
    )
    comparison_results: ComparisonImpressionMetrics = Field(
        description="Impression comparison metrics including CMC counts, fractions, and consensus registration.",
    )

    @computed_field
    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @computed_field
    @property
    def cmc_fraction(self) -> float:
        return self.comparison_results.score / len(self.cells)

    @model_validator(mode="after")
    def score_cannot_exceed_n_cells(self) -> Self:
        """Check that the number of matching cells is equal or smaller than the total number of cells."""
        if self.comparison_results.score > len(self.cells):
            raise ValueError(f"score ({self.comparison_results.score}) cannot exceed n_cells ({len(self.cells)})")
        return self


class ComparisonResponseStriationURL(ComparisonResponse):
    mark_reference_aligned_surface_map: HttpUrl = Field(
        ...,
        description="Surface map data of the reference mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_reference_aligned_surface_map.png"
        ],
    )
    mark_compared_aligned_surface_map: HttpUrl = Field(
        ...,
        description="Surface map data of the compared mark.",
        examples=[
            "http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark_compared_aligned_surface_map.png"
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
    comparison_results: StriationComparisonResults = Field(
        description="Striation comparison metrics including correlation, roughness, and alignment geometry.",
    )


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

    @model_validator(mode="after")
    def check_ci_bounds(self) -> Self:
        """Ensure the confidence interval, if present, actually contains the point estimate."""
        if self.llr_lower_ci is not None and self.llr_lower_ci > self.llr:
            raise ValueError(f"llr_lower_ci ({self.llr_lower_ci}) cannot exceed llr ({self.llr})")
        if self.llr_upper_ci is not None and self.llr_upper_ci < self.llr:
            raise ValueError(f"llr_upper_ci ({self.llr_upper_ci}) cannot be less than llr ({self.llr})")
        return self
