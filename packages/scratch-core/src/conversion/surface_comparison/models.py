from pydantic import Field, field_validator
from collections.abc import Sequence
import numpy as np
from dataclasses import dataclass
from container_models.base import ConfigBaseModel, FloatArray1D, FloatArray2D, Point2D


class CellMetaData(ConfigBaseModel):
    is_outlier: bool
    residual_angle_deg: float = Field(ge=-180, le=180)
    position_error: Point2D


class Cell(ConfigBaseModel):
    """
    :param center_reference: Cell center on reference image [x, y] in meters, shape(2, ).
    :param cell_data: Height_data, in meters, FloatArray2D
    :param fill_fraction_reference: Surface based on the number of pixels divided by the desired surface.
    :param best_score: Best cross correlation score
    :param angle_reference: Angle rotation in degrees corresponding to the correlation score
    :param center_comparison: Cell center on comparison image (x, y) in meters corresponding to the correlation score
    :param is_congruent: True if this cell is classified as a Congruent Matching Cell (CMC).
    """

    # TODO: use tuples instead FloatArray1D?
    center_reference: FloatArray1D
    cell_data: FloatArray2D
    fill_fraction_reference: float = Field(ge=0.0, le=1.0)
    best_score: float = Field(ge=0.0, le=1.0)
    angle_deg: float = Field(ge=-180, le=180)
    center_comparison: FloatArray1D
    is_congruent: bool
    meta_data: CellMetaData

    @field_validator("fill_fraction_reference", "best_score", mode="before")
    @classmethod
    def check_upper_bound_with_tol(cls, value: float | None):
        tol = 1e-6
        if value is None:
            return value
        if value > 1.0 + tol:
            raise ValueError(f"value must be ≤ 1.0 (+{tol} tolerance)")
        return min(value, 1.0)  # clip value


@dataclass
class ComparisonResult:
    """
    Consolidated results of the CMC pipeline.

    :param cells: Per-cell registration and classification results.
    :param consensus_rotation: Rotation consensus across CMC cells (degrees).
    :param consensus_translation: Translation consensus across CMC cells (m), shape (2,).
    """

    # TODO: use tuples instead FloatArray1D
    cells: Sequence[Cell]
    consensus_rotation: float
    consensus_translation: FloatArray1D  # shape (2,)

    @property
    def cell_count(self) -> int:
        """Total number of cells."""
        return len(self.cells)

    @property
    def cmc_count(self) -> int:
        """Return total number of CMC's"""
        return sum(c.is_congruent for c in self.cells)

    @property
    def cmc_fraction(self) -> float:
        """Fraction of cells classified as CMC."""
        return self.cmc_count / self.cell_count

    @property
    def cmc_area_fraction(self) -> float:
        """Fraction of valid surface area covered by CMC cells."""
        total_area = sum(cell.fill_fraction_reference for cell in self.cells)
        cmc_area = sum(
            cell.fill_fraction_reference for cell in self.cells if cell.is_congruent
        )
        return cmc_area / total_area


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

    # TODO: Define default values somewhere
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
