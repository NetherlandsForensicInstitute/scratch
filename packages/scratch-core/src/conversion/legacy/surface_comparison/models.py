from pydantic import Field, field_validator

import numpy as np

from container_models.base import ConfigBaseModel, FloatArray1D, FloatArray2D


class Cell(ConfigBaseModel):
    """
    :param center_reference: Cell center on reference image [x, y] in meters, shape(2, ).
    :param cell_data: Height_data, in meters, FloatArray2D
    :param fill_fraction_reference: Surface based on the number of pixels divided by the desired surface.
    :param best_score: best cross_correlation score
    :param angle_reference: angle rotation on reference corresponding to best correlation score
    :param center_comparison: cell center on comparison image (x, y) in meters corresponding to best correlation score
    :param is_congruent: True if this cell is classified as a Congruent Matching Cell (CMC).

    """

    center_reference: FloatArray1D
    cell_data: FloatArray2D
    fill_fraction_reference: float = Field(..., ge=0.0, le=1.0)
    best_score: float | None = Field(None, le=1.0)
    angle_reference: float | None = Field(None, ge=-180, le=180)
    center_comparison: FloatArray1D | None = None
    is_congruent: bool = False

    @field_validator("fill_fraction_reference", "best_score", mode="before")
    @classmethod
    def check_upper_bound_with_tol(cls, v):
        TOL = 1e-6
        if v is None:
            return v
        if v > 1.0 + TOL:
            raise ValueError(f"value must be ≤ 1.0 (+{TOL} tolerance)")
        return min(v, 1.0)  # optionally clip


class ComparisonResult(ConfigBaseModel):
    """
    Consolidated results of the CMC pipeline.

    :param cells: Per-cell registration and classification results.
    :param congruent_matching_cells_count: Number of cells classified as CMC.
    :param consensus_rotation: Rotation consensus across CMC cells (degrees).
    :param consensus_translation: Translation consensus across CMC cells (m), shape (2,).
    """

    cells: list[Cell] = Field(default_factory=list)
    congruent_matching_cells_count: int = 0
    consensus_rotation: float = 0.0
    consensus_translation: FloatArray1D = Field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )

    # ComparisonResult is intentionally mutable: the pipeline populates cells
    # and updates counts after construction.
    model_config = {
        **ConfigBaseModel.model_config,
        "frozen": False,
    }

    @property
    def cell_count(self) -> int:
        """Total number of cells."""
        return len(self.cells)

    @property
    def cmc_fraction(self) -> float:
        """Fraction of cells classified as CMC."""
        return (
            self.congruent_matching_cells_count / self.cell_count
            if self.cells
            else float("nan")
        )

    @property
    def cmc_area_fraction(self) -> float:
        """Fraction of valid surface area covered by CMC cells."""
        total_area = sum(cell.fill_fraction_reference for cell in self.cells)
        if total_area == 0:
            return float("nan")
        cmc_area = sum(
            cell.fill_fraction_reference for cell in self.cells if cell.is_congruent
        )
        return cmc_area / total_area

    def update_summary(self) -> None:
        """Recount CMC cells from current cell statuses."""
        self.congruent_matching_cells_count = sum(
            1 for cell in self.cells if cell.is_congruent
        )


class ComparisonParams(ConfigBaseModel):
    """
    Parameters for the Congruent Matching Cells (CMC) algorithm.

    :param cell_size: Nominal cell size [width, height] in meters, shape (2,).
    :param minimum_fill_fraction: Minimum fraction of valid pixels required in a
        reference cell for it to be processed.
    :param correlation_threshold: Minimum per-cell ACCF score for CMC classification.
    :param angle_threshold: Maximum angular deviation from consensus for CMC (degrees).
    :param position_threshold: Maximum positional deviation from consensus for CMC (m).
    :param search_angle_min: Lower bound of rotation search range (degrees).
    :param search_angle_max: Upper bound of rotation search range (degrees).
    :param search_angle_step: Angular step size for the coarse rotation sweep (degrees).
    """

    cell_size: FloatArray1D = Field(
        default_factory=lambda: np.array([1e-3, 1e-3], dtype=np.float64)
    )
    minimum_fill_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    correlation_threshold: float = Field(default=0.4, ge=-1.0, le=1.0)
    angle_threshold: float = Field(default=2.0, gt=0.0)
    position_threshold: float = Field(default=100e-6, gt=0.0)
    search_angle_min: float = -180.0
    search_angle_max: float = 180.0
    search_angle_step: float = Field(default=1.0, gt=0.0)
