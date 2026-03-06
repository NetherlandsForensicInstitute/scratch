import datetime
from typing import Self

from conversion.plots.data_formats import ImpressionComparisonMetrics
from pydantic import DirectoryPath, Field, FilePath, NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator

from models import BaseModelConfig


class MarkDirectories(BaseModelConfig):
    mark_dir_ref: DirectoryPath
    mark_dir_comp: DirectoryPath

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return "SomethingWithNoValue"


class ImpressionParameters(BaseModelConfig): ...


class CalculateScoreImpression(MarkDirectories):
    param: ImpressionParameters


class StriationParameters(BaseModelConfig):
    metadata_reference: dict[str, str] = Field(..., description="fields needed for adding metadata to the plot")
    metadata_compared: dict[str, str] = Field(..., description="fields needed for adding metadata to the plot")


class CalculateScoreStriation(MarkDirectories):
    param: StriationParameters


class CalculateLR(MarkDirectories):
    lr_system_path: FilePath
    user_id: str
    date_report: datetime.date


class ImpressionLRParameters(BaseModelConfig): ...


class CalculateLRImpression(CalculateLR, StriationParameters):
    score: NonNegativeInt
    n_cells: PositiveInt
    param: ImpressionLRParameters
    metrics: ImpressionComparisonMetrics

    @model_validator(mode="after")
    def score_cannot_exceed_n_cells(self) -> Self:
        """Ensure that the score <= n_cells."""
        if self.score > self.n_cells:
            raise ValueError(f"score ({self.score}) cannot exceed n_cells ({self.n_cells})")
        return self


class StriationLRParameters(BaseModelConfig): ...


class CalculateLRStriation(CalculateLR, StriationParameters):
    mark_dir_ref_aligned: DirectoryPath
    mark_dir_comp_aligned: DirectoryPath
    score: NonNegativeFloat
    param: StriationLRParameters
