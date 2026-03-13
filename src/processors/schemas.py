import datetime
from collections.abc import Sequence
from typing import Annotated, Self

from conversion.data_formats import MarkMetadata
from conversion.surface_comparison.models import Cell, ComparisonParams
from pydantic import DirectoryPath, Field, FilePath, NonNegativeInt, PositiveInt, model_validator

from models import BaseModelConfig


class MarkDirectories(BaseModelConfig):
    mark_dir_ref: DirectoryPath
    mark_dir_comp: DirectoryPath

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return "SomethingWithNoValue"


class MetadataParameters(BaseModelConfig):
    metadata_reference: MarkMetadata = Field(..., description="Metadata identifying the reference mark.")
    metadata_compared: MarkMetadata = Field(..., description="Metadata identifying the compared mark.")


class CalculateScore(MarkDirectories, MetadataParameters): ...


class CalculateScoreImpression(CalculateScore):
    comparison_params: ComparisonParams


class CalculateLR(MarkDirectories, MetadataParameters):
    lr_system_path: FilePath
    user_id: str
    date_report: datetime.date


class CalculateLRImpression(CalculateLR):
    score: NonNegativeInt
    n_cells: PositiveInt
    cells: Sequence[Cell]

    @model_validator(mode="after")
    def score_cannot_exceed_n_cells(self) -> Self:
        """Ensure that the score <= n_cells."""
        if self.score > self.n_cells:
            raise ValueError(f"score ({self.score}) cannot exceed n_cells ({self.n_cells})")
        return self


class CalculateLRStriation(CalculateLR):
    mark_dir_ref_aligned: DirectoryPath
    mark_dir_comp_aligned: DirectoryPath
    score: Annotated[float, Field(ge=-1, le=1)]

    @property
    def tag(self) -> str:
        # TODO: incorporate mark_dir_ref_aligned and mark_dir_comp_aligned once
        # the base tag implementation is finalised.
        return super().tag
