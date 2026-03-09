import datetime
from functools import cached_property
from typing import Self

import numpy as np
from container_models.base import FloatArray1D, FloatArray2D
from pydantic import DirectoryPath, Field, FilePath, NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator

from models import BaseModelConfig


class MarkDirectories(BaseModelConfig):
    mark_dir_ref: DirectoryPath
    mark_dir_comp: DirectoryPath

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return "SomethingWithNoValue"


class ImpressionParameters(BaseModelConfig):
    metadata_reference: dict[str, str] = Field(..., description="fields needed for adding metadata to the plot")
    metadata_compared: dict[str, str] = Field(..., description="fields needed for adding metadata to the plot")


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


class ImpressionLRParameters(BaseModelConfig):
    area_correlation: float
    cell_correlations: list[list[float]]
    cmc_score: float
    mean_square_ref: float
    mean_square_comp: float
    mean_square_of_difference: float
    has_area_results: bool
    has_cell_results: bool
    cell_positions_compared: list[list[float]]
    cell_rotations_compared: list[float]
    cmc_area_fraction: float
    cutoff_low_pass: float
    cutoff_high_pass: float
    cell_size_um: float
    max_error_cell_position: float
    max_error_cell_angle: float
    cell_similarity_threshold: float = 0.25

    @cached_property
    def cell_correlations_array(self) -> FloatArray2D:
        return np.array(self.cell_correlations)

    @cached_property
    def cell_positions_compared_array(self) -> FloatArray2D:
        return np.array(self.cell_positions_compared)

    @cached_property
    def cell_rotations_compared_array(self) -> FloatArray1D:
        return np.array(self.cell_rotations_compared)


class CalculateLRImpression(CalculateLR, StriationParameters):
    score: NonNegativeInt
    n_cells: PositiveInt
    param: ImpressionLRParameters

    @model_validator(mode="after")
    def score_cannot_exceed_n_cells(self) -> Self:
        """Ensure that the score <= n_cells."""
        if self.score > self.n_cells:
            raise ValueError(f"score ({self.score}) cannot exceed n_cells ({self.n_cells})")
        return self


class CalculateLRStriation(CalculateLR, StriationParameters):
    mark_dir_ref_aligned: DirectoryPath
    mark_dir_comp_aligned: DirectoryPath
    score: NonNegativeFloat
