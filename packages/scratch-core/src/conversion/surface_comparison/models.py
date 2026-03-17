from functools import cached_property

import numpy as np
from pydantic import Field, field_validator, PositiveFloat

from collections.abc import Sequence
from dataclasses import dataclass

from scipy.constants import mega

from container_models.base import ConfigBaseModel, FloatArray2D
from conversion.data_formats import Mark


@dataclass(frozen=True)
class ProcessedMark:
    """Container class for storing processed `Mark` instances."""

    filtered_mark: Mark
    leveled_mark: Mark


class CellMetaData(ConfigBaseModel):
    """
    Intermediate classification data computed during the CMC pipeline.

    :param is_outlier: True if this cell was rejected as an angle outlier during
        consensus estimation (ESD test or tightening step).
    :param residual_angle_deg: Signed angular deviation from the consensus rotation,
        in degrees, after the final inlier median is computed.
    :param position_error: Signed [x, y] deviation from the consensus translation,
        in meters.
    """

    is_outlier: bool
    residual_angle_deg: float = Field(ge=-180, le=180)
    position_error: tuple[float, float]


class Cell(ConfigBaseModel):
    """
    Per-cell registration result and CMC classification outcome.

    :param center_reference: Cell center on the reference image [x, y] in meters.
    :param cell_size: Cell size on the reference image [width, height] in meters.
    :param fill_fraction_reference: Fraction of valid pixels in this cell relative
        to the nominal cell area (0 = empty, 1 = fully filled).
    :param best_score: Best ACCF cross-correlation score achieved for this cell.
    :param angle_deg: Rotation angle in degrees for the reference image at which the best score was obtained.
    :param center_comparison: Cell center on the comparison image [x, y] in meters
        at which the best score was obtained.
    :param is_congruent: True if this cell is classified as a Congruent Matching Cell.
    :param meta_data: Intermediate pipeline data (outlier flag, angle residual,
        position error) populated by the classifier.
    """

    center_reference: tuple[float, float]
    cell_size: tuple[float, float]
    fill_fraction_reference: float = Field(ge=0.0, le=1.0)
    best_score: float = Field(ge=0.0, le=1.0)
    angle_deg: float = Field(ge=-180, le=180)
    center_comparison: tuple[float, float]
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

    @property
    def cell_size_um(self) -> tuple[float, float]:
        return self.cell_size[0] * mega, self.cell_size[1] * mega


@dataclass
class ComparisonResult:
    """
    Consolidated results of the CMC pipeline.

    :param cells: Per-cell registration and classification results.
    :param shared_rotation: Rotation consensus across CMC cells (degrees).
    :param shared_translation: Translation consensus across CMC cells (m)
    """

    cells: Sequence[Cell]
    shared_rotation: float
    shared_translation: tuple[float, float]

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

    :param cell_size: Nominal cell size [width, height] in meters.
    :param minimum_fill_fraction: Minimum fraction of valid pixels required in a
        reference cell for it to be processed.
    :param correlation_threshold: Minimum per-cell ACCF score for CMC classification.
    :param angle_deviation_threshold: Maximum absolute angular deviation from consensus for CMC (degrees).
    :param position_threshold: Maximum positional deviation from consensus for CMC (m).
    :param search_angle_min: Lower bound of rotation search range (degrees).
    :param search_angle_max: Upper bound of rotation search range (degrees).
    :param search_angle_step: Angular step size for the coarse rotation sweep (degrees).
    """

    cell_size: tuple[PositiveFloat, PositiveFloat] = (1e-3, 1e-3)
    minimum_fill_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    correlation_threshold: float = Field(default=0.3, ge=-1.0, le=1.0)
    angle_deviation_threshold: float = Field(default=2.0, gt=0.0)
    position_threshold: float = Field(default=100e-6, gt=0.0)
    search_angle_min: float = -180.0
    search_angle_max: float = 180.0
    search_angle_step: float = Field(default=1.0, gt=0.0)


@dataclass(frozen=False)
class GridSearchParams:
    """
    Mutable container for the best registration parameters found so far for one cell.
    All positional attributes are in pixel coordinates of the (rotated) comparison image.
    :param center_x: Center x-coordinate of the best-matching comparison patch (pixels).
    :param center_y: Center y-coordinate of the best-matching comparison patch (pixels).
    :param angle: Rotation angle at which the best score was found (degrees).
    :param score: Best normalized cross-correlation score found so far.
    """

    center_x: float = -1.0
    center_y: float = -1.0
    angle: float = 0.0
    score: float = float("-inf")

    def update(
        self, center_x: float, center_y: float, angle: float, score: float
    ) -> None:
        """Replace all fields with a new best result."""
        self.center_x = center_x
        self.center_y = center_y
        self.angle = angle
        self.score = score


@dataclass(frozen=True)
class GridCell:
    """
    Container class for storing generated grid cells.

    All the values of the attributes and properties are in pixel units.

    :param top_left: Tuple containing the top-left pixel coordinates (x, y) corresponding to the reference image.
    :param cell_data: 2D array containing the sliced image data from the reference image.
    :param grid_search_params: An instance of `GridSearchParams` for keeping track of intermediate search results.
    :param nan_fill_value: (Optional) A sentinel value for replacing NaN values. The cell data with the NaN
        values replaced is stored in the `cell_data_filled` attribute.
    """

    top_left: tuple[int, int]
    cell_data: FloatArray2D
    grid_search_params: GridSearchParams
    nan_fill_value: float = np.nan

    @property
    def width(self) -> int:
        return self.cell_data.shape[1]

    @property
    def height(self) -> int:
        return self.cell_data.shape[0]

    @property
    def center(self) -> tuple[float, float]:
        return self.top_left[0] + self.width / 2, self.top_left[1] + self.height / 2

    @property
    def fill_fraction(self) -> float:
        return float(np.count_nonzero(~np.isnan(self.cell_data)) / self.cell_data.size)

    @cached_property
    def cell_data_filled(self) -> FloatArray2D:
        """Cell data where NaN values are replaced with the sentinel value."""
        return np.nan_to_num(self.cell_data, nan=self.nan_fill_value, copy=True)
