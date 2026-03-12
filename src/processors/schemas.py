import datetime
from typing import Annotated, Self

import numpy as np
from container_models.base import FloatArray1D, FloatArray2D
from conversion.data_formats import MarkMetadata
from conversion.plots.data_formats import ImpressionComparisonMetrics
from conversion.surface_comparison.models import ComparisonParams
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


class CalculateScore(MarkDirectories):
    param: MetadataParameters


class CalculateScoreImpression(CalculateScore):
    comparison_params: ComparisonParams


class CalculateLR(MarkDirectories, MetadataParameters):
    lr_system_path: FilePath
    user_id: str
    date_report: datetime.date


class ImpressionLRParameters(BaseModelConfig):
    """Impression comparison metrics as JSON-serialisable fields.

    FastAPI cannot deserialise numpy arrays directly from JSON, so this class
    mirrors :class:`~conversion.plots.data_formats.ImpressionComparisonMetrics`
    using plain Python lists. The ``*_array`` properties convert to numpy on
    access. Update this class whenever ``ImpressionComparisonMetrics`` changes.
    """

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

    def to_metrics(self) -> ImpressionComparisonMetrics:
        """Convert to ImpressionComparisonMetrics with numpy arrays."""
        return ImpressionComparisonMetrics(
            cell_correlations=self.cell_correlations_array,
            cmc_score=self.cmc_score,
            cell_positions_compared=self.cell_positions_compared_array,
            cell_rotations_compared=self.cell_rotations_compared_array,
            cmc_area_fraction=self.cmc_area_fraction,
            cell_size_um=self.cell_size_um,
            max_error_cell_position=self.max_error_cell_position,
            max_error_cell_angle=self.max_error_cell_angle,
            cell_similarity_threshold=self.cell_similarity_threshold,
        )


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
    score: Annotated[float, Field(ge=-1, le=1)]

    @property
    def tag(self) -> str:
        # TODO: incorporate mark_dir_ref_aligned and mark_dir_comp_aligned once
        # the base tag implementation is finalised.
        return super().tag
