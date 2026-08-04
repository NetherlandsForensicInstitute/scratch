from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import numpy as np
from pydantic import Field, PositiveFloat, field_validator
from scipy.constants import mega

from container_models.base import ConfigBaseModel, FloatArray2D
from conversion.data_formats import Mark, MarkImpressionType


@dataclass(frozen=True)
class ProcessedMark:
    filtered_mark: Mark
    raw_mark: Mark


class CellMetaData(ConfigBaseModel):
    is_outlier: bool
    residual_angle_deg: float = Field(ge=-180, le=180)
    position_error: tuple[float, float] = Field(..., examples=[-9.12, 6.8])


class Cell(ConfigBaseModel):
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


@dataclass
class ComparisonResult:
    cells: Sequence[Cell]
    estimated_rotation: float
    estimated_translation: tuple[float, float]

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def cmc_count(self) -> int:
        return sum(c.is_congruent for c in self.cells)

    @property
    def cmc_fraction(self) -> float:
        return self.cmc_count / self.cell_count

    @property
    def cmc_area_fraction(self) -> float:
        total_area = sum(cell.fill_fraction_reference for cell in self.cells)
        cmc_area = sum(
            cell.fill_fraction_reference for cell in self.cells if cell.is_congruent
        )
        return cmc_area / total_area


_CELL_SIZE_BY_MARK_TYPE: dict[MarkImpressionType, tuple[float, float]] = {
    MarkImpressionType.BREECH_FACE_IMPRESSION: (4.5e-4, 4.5e-4),
    MarkImpressionType.CHAMBER_IMPRESSION: (1.25e-4, 1.25e-4),
    MarkImpressionType.EJECTOR_IMPRESSION: (1.25e-4, 1.25e-4),
    MarkImpressionType.EXTRACTOR_IMPRESSION: (1.25e-4, 1.25e-4),
    MarkImpressionType.FIRING_PIN_IMPRESSION: (1.25e-4, 1.25e-4),
}


class ComparisonParams(ConfigBaseModel):
    cell_size: tuple[PositiveFloat, PositiveFloat] = (4.5e-4, 4.5e-4)

    @classmethod
    def for_mark_type(
        cls, mark_type: MarkImpressionType, **kwargs: Any
    ) -> "ComparisonParams":
        if mark_type not in _CELL_SIZE_BY_MARK_TYPE:
            raise ValueError(
                f"No default cell size registered for mark type: {mark_type!r}"
            )
        return cls(cell_size=_CELL_SIZE_BY_MARK_TYPE[mark_type], **kwargs)

    minimum_fill_fraction: float = Field(default=0.35, ge=0.0, le=1.0)
    correlation_threshold: float = Field(default=0.25, ge=-1.0, le=1.0)
    angle_deviation_threshold: float = Field(default=6.0, gt=0.0)
    position_threshold: float = Field(default=7.5e-5, gt=0.0)
    search_angle_min: float = -180.0
    search_angle_max: float = 180.0
    search_angle_step: float = Field(default=5.0, gt=0.0)

    # --- Coarse stage: exhaustive translation + rotation sweep on a downsampled image pair ---
    max_size: int = Field(
        default=1000,
        gt=0,
        description=(
            "Largest permitted dimension (pixels) of the comparison canvas used for the coarse "
            "exhaustive sweep. Both images are downsampled together, once, so they still share a "
            "pixel scale; images already at or below this size are left alone."
        ),
    )
    downsample_interpolation: Literal["area", "linear", "nearest", "cubic"] = Field(
        default="area",
        description=(
            "Interpolation used whenever an image is downsampled (pixel-scale alignment and the "
            "coarse-stage size cap): one of 'area', 'linear', 'nearest', 'cubic'. 'area' is the "
            "recommended default for shrinking images; the others are exposed to make it easy to "
            "empirically compare algorithms on real data instead of assuming one is better."
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

    # --- Fine stage: local search around each coarse candidate, on the original-resolution images ---
    fine_n_pixels: int = Field(
        default=5,
        ge=0,
        description="Fine-stage translation margin: search ±N pixels around each candidate's position.",
    )
    fine_m_degrees: float = Field(
        default=5.0,
        ge=0.0,
        description="Fine-stage angle margin: search ±M degrees, in 1-degree steps, around each candidate's angle.",
    )
    fine_batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Refinement jobs (candidate pose x trial angle) processed per chunk. None picks a device-based default.",
    )


@dataclass(frozen=False)
class GridSearchParams:
    center_x: float = -1.0
    center_y: float = -1.0
    angle: float = 0.0
    score: float = float("-inf")

    def update(
        self, center_x: float, center_y: float, angle: float, score: float
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.angle = angle
        self.score = score


@dataclass(frozen=True)
class GridCell:
    top_left: tuple[int, int]
    cell_data: FloatArray2D
    grid_search_params: GridSearchParams

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
        Cell data with NaNs filled by this cell's *own* valid-pixel mean, not a scene-wide value.

        The registration search centers every template on its own mean before correlating (see
        ``_prepare_templates`` in ``cell_registration_utils.py``). Filling with the cell's own mean
        means every originally-missing pixel becomes exactly zero once centered, so it contributes
        nothing to the correlation sum, the template's variance, or its norm: a missing pixel is
        treated as "no information", rather than as a real, flat patch of surface. Filling with an
        unrelated global value (e.g. the whole reference image's mean) does not have this property
        and biases cells with a lower fill fraction.
        """
        local_mean = np.nanmean(self.cell_data)
        fill_value = float(local_mean) if np.isfinite(local_mean) else 0.0
        return np.nan_to_num(self.cell_data, nan=fill_value, copy=True)
