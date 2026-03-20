import datetime
from collections.abc import Sequence
from typing import Annotated, Self

from conversion.data_formats import MarkMetadata
from conversion.surface_comparison.models import Cell, ComparisonParams
from pydantic import DirectoryPath, Field, FilePath, NonNegativeInt, PositiveInt, model_validator

from models import BaseModelConfig


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
    lr_system_path: FilePath = Field(
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
    n_cells: PositiveInt = Field(
        ...,
        description="Total number of cells in the comparison grid.",
    )
    cells: Sequence[Cell] = Field(
        ...,
        description="Per-cell CMC results from the impression comparison.",
    )

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
