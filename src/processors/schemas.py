import datetime
from typing import Self

import numpy as np
from container_models.base import FloatArray1D, FloatArray2D
from conversion.data_formats import MarkMetadata
from pydantic import DirectoryPath, Field, FilePath, NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator

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


class CalculateScoreImpression(MarkDirectories):
    param: MetadataParameters


class CalculateScoreStriation(MarkDirectories):
    param: MetadataParameters


class CalculateLR(MarkDirectories):
    lr_system_path: FilePath
    user_id: str
    date_report: datetime.date
    metadata_reference: MarkMetadata
    metadata_compared: MarkMetadata


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

    @property
    def cell_correlations_array(self) -> FloatArray2D:
        """Return cell correlations as a 2D numpy array."""
        return np.array(self.cell_correlations)

    @property
    def cell_positions_compared_array(self) -> FloatArray2D:
        """Return compared cell positions as a 2D numpy array."""
        return np.array(self.cell_positions_compared)

    @property
    def cell_rotations_compared_array(self) -> FloatArray1D:
        """Return compared cell rotations as a 1D numpy array."""
        return np.array(self.cell_rotations_compared)


class CalculateLRImpression(CalculateLR):
    score: NonNegativeInt
    n_cells: PositiveInt
    param: ImpressionLRParameters

    @model_validator(mode="after")
    def score_cannot_exceed_n_cells(self) -> Self:
        """Ensure that the score <= n_cells."""
        if self.score > self.n_cells:
            raise ValueError(f"score ({self.score}) cannot exceed n_cells ({self.n_cells})")
        return self


class CalculateLRStriation(CalculateLR):
    mark_dir_ref_aligned: DirectoryPath
    mark_dir_comp_aligned: DirectoryPath
    score: NonNegativeFloat
