from typing import Self

from pydantic import DirectoryPath, Field, FilePath, PositiveInt, model_validator

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


class ImpressionLRParameters(BaseModelConfig): ...


class CalculateLRImpression(CalculateLR):
    score: int
    n_cells: PositiveInt
    param: ImpressionLRParameters

    @model_validator(mode="after")
    def score_cannot_exceed_n_cells(self) -> Self:
        """Ensure that the score <= n_cells."""
        if self.score > self.n_cells:
            raise ValueError(f"score ({self.score}) cannot exceed n_cells ({self.n_cells})")
        return self


class StriationLRParameters(BaseModelConfig): ...


class CalculateLRStriation(CalculateLR):
    score: float
    param: StriationLRParameters
