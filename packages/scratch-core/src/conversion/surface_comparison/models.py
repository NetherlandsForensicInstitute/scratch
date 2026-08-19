from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from textwrap import dedent
from typing import Literal

import numpy as np
from pydantic import Field, field_validator
from scipy.constants import mega

from container_models.base import ConfigBaseModel, FloatArray2D
from conversion.data_formats import Mark
from conversion.surface_comparison.template_fill import fill_template_nan


@dataclass(frozen=True)
class ProcessedMark:
    """Container class for storing processed `Mark` instances."""

    filtered_mark: Mark
    raw_mark: Mark


class CellMetaData(ConfigBaseModel):
    """
    Intermediate classification data computed during the CMC pipeline.

    :param is_outlier: True if this cell was rejected as an angle outlier during consensus estimation.
    :param residual_angle_deg: Signed angular deviation from the consensus rotation, in degrees.
    :param position_error: Signed [x, y] deviation from the consensus translation, in meters.
    """

    is_outlier: bool
    residual_angle_deg: float = Field(ge=-180, le=180)
    position_error: tuple[float, float] = Field(..., examples=[-9.12, 6.8])


class Cell(ConfigBaseModel):
    """
    Per-cell registration result and CMC classification outcome.

    :param center_reference: Cell center on the reference image [x, y] in meters.
    :param cell_size: Cell size on the reference image [width, height] in meters.
    :param fill_fraction_reference: Fraction of valid pixels relative to nominal area (0 = empty, 1 = fully filled).
    :param best_score: Best ACCF cross-correlation score achieved.
    :param angle_deg: Rotation angle in degrees for the reference image at the best score.
    :param center_comparison: Cell center on the comparison image [x, y] in meters at the best score.
    :param is_congruent: True if this cell is classified as a Congruent Matching Cell.
    :param meta_data: Intermediate pipeline data (outlier flag, angle residual, position error).
    """

    center_reference: tuple[float, float] = Field(..., examples=[(4.5, 1.4)])
    cell_size: tuple[float, float] = Field(..., examples=[(2.1, 1.90)])
    fill_fraction_reference: float = Field(ge=0.0, le=1.0)
    best_score: float = Field(ge=-1.0, le=1.0)
    angle_deg: float = Field(ge=-180, le=180)
    center_comparison: tuple[float, float] = Field(..., examples=[(5.6, 7.4)])
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
        return min(value, 1.0)

    @field_validator("angle_deg", mode="before")
    @classmethod
    def wrap_angle_deg(cls, value: float | None) -> float | None:
        if value is None:
            return value
        # Wrap -181.0 -> 179.0, 181.0 -> -179.0
        return (float(value) + 180.0) % 360.0 - 180.0

    @property
    def cell_size_um(self) -> tuple[float, float]:
        return self.cell_size[0] * mega, self.cell_size[1] * mega


# Sentinel pose reported when there is no consensus geometry to estimate one from.
# NaN matches the median classifier; the rotation cannot be NaN because the API schema bounds it.
NO_CONSENSUS_ROTATION = 0.0
NO_CONSENSUS_TRANSLATION = (float("nan"), float("nan"))


@dataclass
class ComparisonResult:
    """
    Consolidated results of the CMC pipeline.

    :param cells: Per-cell registration and classification results.
    :param estimated_rotation: Estimated rotation across CMC cells (degrees).
    :param estimated_translation: Estimated translation across CMC cells (m)
    """

    cells: Sequence[Cell]
    estimated_rotation: float
    estimated_translation: tuple[float, float]

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

    :param minimum_fill_fraction: Minimum fraction of valid pixels required in a reference cell for it to be processed.
    :param correlation_threshold: Minimum per-cell ACCF score for CMC classification.
    :param angle_deviation_threshold: Maximum absolute angular deviation from consensus for CMC (degrees).
    :param position_threshold: Maximum positional deviation from consensus for CMC (m).
    :param search_angle_min: Lower bound of rotation search range (degrees).
    :param search_angle_max: Upper bound of rotation search range (degrees).
    :param search_angle_step: Angular step size for the coarse rotation sweep (degrees).

    Remaining fields configure search stages. Image resampling is fixed in `conversion.surface_comparison.pipeline`.
    """

    minimum_fill_fraction: float = Field(default=0.35, ge=0.0, le=1.0)
    correlation_threshold: float = Field(default=0.25, ge=-1.0, le=1.0)
    cmc_algorithm: Literal["median", "consensus"] = Field(
        default="consensus",
        description="CMC classification algorithm: 'consensus' (pairwise Procrustes) "
        "or 'median' (median-based with ESD).",
    )
    angle_deviation_threshold: float = Field(default=6.0, gt=0.0)
    position_threshold: float = Field(default=7.5e-5, gt=0.0)
    search_angle_min: float = -180.0
    search_angle_max: float = 180.0
    search_angle_step: float = Field(default=5.0, gt=0.0)

    coarse_target_size: int = Field(
        default=256,
        gt=0,
        description=(
            "Target image size in pixels for the coarse sweep; a minimum cell size can keep it larger."
        ),
    )
    n_candidates: int = Field(
        default=3,
        ge=1,
        description="Candidate (x, y, angle) poses kept per cell from the coarse stage, for refinement.",
    )
    angle_batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Angles processed per chunk during the coarse sweep. None picks a device-based default.",
    )
    template_batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Cells processed per chunk during the coarse sweep. None picks a device-based default.",
    )

    fine_n_pixels: int = Field(
        default=16,
        ge=0,
        description="Fine-stage translation margin: search ±N pixels around each candidate's position.",
    )
    fine_m_degrees: float = Field(
        default=10.0,
        ge=0.0,
        description="Fine-stage angle margin: search ±M degrees, in 1-degree steps, around each candidate's angle.",
    )
    fine_batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Refinement jobs (candidate pose x trial angle) processed per chunk. None picks a device-based default.",
    )

    template_nan_fill_strategy: Literal["local_mean", "global_mean"] = Field(
        default="global_mean",
        description=dedent("""
            How NaN pixels in reference templates are filled before correlation.
            - local_mean: each cell's own valid-pixel mean; filled pixels contribute nothing.
            - global_mean: the reference image's global mean; filled pixels count as flat surface.
            """),
    )


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
        """
        Replace all fields with a new best result.

        :param center_x: Center x-coordinate of the new best-matching comparison patch (pixels).
        :param center_y: Center y-coordinate of the new best-matching comparison patch (pixels).
        :param angle: Rotation angle at which the new best score was found (degrees).
        :param score: New best normalized cross-correlation score.
        """
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
    :param nan_fill_value: Optional fill value for NaN pixels. When provided, NaN are replaced with
        this value; when None, each cell's own valid-pixel mean is used.
    """

    top_left: tuple[int, int]
    cell_data: FloatArray2D
    grid_search_params: GridSearchParams
    nan_fill_value: float | None = None

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
        """
        Cell data with NaN filled per fill_template_nan.

        ``nan_fill_value`` carries the resolved ``template_nan_fill_strategy``: an explicit value
        for ``global_mean``, ``None`` for ``local_mean``.
        """
        return fill_template_nan(self.cell_data, self.nan_fill_value)
