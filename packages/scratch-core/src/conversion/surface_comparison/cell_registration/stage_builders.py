"""Cell registration: stage builders and result recording."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.resample import resample_nan_aware
from conversion.surface_comparison.cell_registration.geometry import (
    map_canvas_to_image,
    pad_image_array,
)
from conversion.surface_comparison.cell_registration.models import Match
from conversion.surface_comparison.grid import extract_patch
from conversion.surface_comparison.models import (
    Cell,
    CellMetaData,
    ComparisonParams,
    GridCell,
)
from conversion.surface_comparison.template_fill import fill_template_nan
from conversion.surface_comparison.utils import convert_pixels_to_meters

#: Minimum coarse cell size for reliable matching. If downsampling would produce cells smaller
#: than this, the cap factor is reduced to keep coarse cells above this threshold.
_MIN_COARSE_CELL = 12


@dataclass(frozen=True)
class Stage:
    """
    Data for one matching stage.

    :param image: Padded comparison canvas we search *in*.
    :param templates: Reference templates we search *for* (one per grid cell).
    :param fill_value: NaN fill value for the comparison image.
    """

    image: FloatArray2D
    templates: list[FloatArray2D]
    fill_value: float


def convert_grid_cell_to_cell(grid_cell: GridCell, pixel_size: float) -> Cell:
    """
    Convert a grid cell's registration result to a Cell in meters.

    :param grid_cell: Grid cell whose search results to convert.
    :param pixel_size: Pixel size in meters (assumed isotropic).
    :returns: A :class:`Cell` with the grid cell's registration data expressed in meters.
    """
    return Cell(
        center_reference=convert_pixels_to_meters(
            values=grid_cell.center, pixel_size=pixel_size
        ),
        cell_size=convert_pixels_to_meters(
            values=(grid_cell.width, grid_cell.height), pixel_size=pixel_size
        ),
        fill_fraction_reference=grid_cell.fill_fraction,
        best_score=grid_cell.grid_search_params.score,
        angle_deg=grid_cell.grid_search_params.angle,
        center_comparison=convert_pixels_to_meters(
            values=(
                grid_cell.grid_search_params.center_x,
                grid_cell.grid_search_params.center_y,
            ),
            pixel_size=pixel_size,
        ),
        is_congruent=False,  # TODO: We shouldn't set this here?
        meta_data=CellMetaData(
            is_outlier=False, residual_angle_deg=0.0, position_error=(0, 0)
        ),  # TODO: We shouldn't set this here?
    )


def build_angle_sweep(params: ComparisonParams) -> np.ndarray:
    """
    Build the coarse stage's angle sweep in degrees (inclusive of both bounds).

    :param params: Comparison parameters defining the angle range and step.
    :returns: Array of angles from ``search_angle_min`` to ``search_angle_max`` in steps of
        ``search_angle_step``.
    """
    return np.arange(
        params.search_angle_min,
        params.search_angle_max + params.search_angle_step,
        params.search_angle_step,
    )


def compute_cap_factor(
    reference_image: ScanImage,
    comparison_image: ScanImage,
    cell_width: int,
    cell_height: int,
    max_size: int,
) -> float:
    """
    Pixels per coarse pixel. Shrinks images to *max_size* while keeping coarse cells
    above :data:`_MIN_COARSE_CELL` pixels for reliable localization.

    :param reference_image: Reference scan image.
    :param comparison_image: Comparison scan image (already on the reference scale).
    :param cell_width: Width of one grid cell in pixels.
    :param cell_height: Height of one grid cell in pixels.
    :param max_size: Largest permitted dimension (pixels) of the comparison canvas.
    :returns: Downsampling factor; 1.0 if images already fit within *max_size* or cells are too
        small to downsample further.
    """
    largest_dimension = max(
        reference_image.height,
        reference_image.width,
        comparison_image.height,
        comparison_image.width,
    )
    raw_cap_factor = max(1.0, largest_dimension / max_size)
    min_allowed_cap = max(1.0, min(cell_width, cell_height) / _MIN_COARSE_CELL)
    cap_factor = min(raw_cap_factor, min_allowed_cap)

    if raw_cap_factor > min_allowed_cap:
        logger.info(
            "Reduced cap factor from {:.2f} to {:.2f}: cells are {}x{} px, "
            "and a coarse cell below {} px cannot localize reliably.",
            raw_cap_factor,
            cap_factor,
            cell_width,
            cell_height,
            _MIN_COARSE_CELL,
        )
    logger.info(
        "Coarse stage config: largest_dim={}, max_size={}, raw_cap={:.2f}, "
        "effective_cap={:.2f}, coarse_stage_runs={}",
        largest_dimension,
        max_size,
        raw_cap_factor,
        cap_factor,
        cap_factor > 1.0,
    )
    return cap_factor


def build_full_resolution_stage(
    comparison_image: ScanImage,
    grid_cells: list[GridCell],
    cell_width: int,
    cell_height: int,
) -> Stage:
    """
    Build the full-resolution stage: padded comparison canvas and filled templates.

    :param comparison_image: Comparison scan image at the reference's pixel scale.
    :param grid_cells: Reference grid cells; templates are taken from their filled cell data.
    :param cell_width: Width of one grid cell in pixels.
    :param cell_height: Height of one grid cell in pixels.
    :returns: A :class:`Stage` with a padded comparison canvas, templates, and fill value.
    """
    return Stage(
        image=pad_image_array(
            comparison_image.data, pad_width=cell_width, pad_height=cell_height
        ),
        templates=[grid_cell.cell_data_filled for grid_cell in grid_cells],
        fill_value=float(np.nanmean(comparison_image.data)),
    )


def build_coarse_stage(
    comparison_image: ScanImage,
    reference_image: ScanImage,
    grid_cells: list[GridCell],
    cap_factor: float,
) -> Stage:
    """
    Downsample both images once, directly to the coarse scale, and re-cut templates from the
    coarse reference.

    *comparison_image* is downsampled by *cap_factor* scaled by its own native pixel size
    relative to the reference's (:func:`compute_scale_match_factor`), so the result lands on
    the same physical coarse grid as *reference_image* in a single pass — regardless of whether
    *comparison_image* is the original scan or one already resampled onto the reference's scale.
    Passing the original avoids chaining two lossy resizes to reach the coarse resolution.

    Templates are extracted from the downsampled reference (not individual patches) so edge
    pixels share context with the comparison canvas.

    :param comparison_image: Comparison scan image, at its own native pixel scale or the
        reference's — either is handled correctly.
    :param reference_image: Reference scan image.
    :param grid_cells: Reference grid cells defining template locations.
    :param cap_factor: Pixels per coarse pixel, in reference-image units.
    :returns: A :class:`Stage` with a downsampled comparison canvas, coarse templates, and fill value.
    """
    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    coarse_width = max(1, int(np.ceil(cell_width / cap_factor)))
    coarse_height = max(1, int(np.ceil(cell_height / cap_factor)))
    logger.info(
        "Coarse stage: {}x{} px cells -> {}x{} px coarse cells (cap={:.2f})",
        cell_width,
        cell_height,
        coarse_width,
        coarse_height,
        cap_factor,
    )

    reference_coarse = ScanImage(
        data=downsample(reference_image.data, cap_factor),
        scale_x=reference_image.scale_x * cap_factor,
        scale_y=reference_image.scale_y * cap_factor,
    )
    scale_match_factor = compute_scale_match_factor(reference_image, comparison_image)
    comparison_coarse = downsample(comparison_image.data, cap_factor * scale_match_factor)

    # All cells carry the same resolved fill value, and the coarse templates must use it too: the
    # coarse stage picks the candidate locations, so filling it differently changes where each cell
    # matches, not just how that match is polished.
    nan_fill_value = grid_cells[0].nan_fill_value
    templates = [
        fill_template_nan(
            extract_patch(
                scan_image=reference_coarse,
                coordinates=(
                    round(grid_cell.top_left[0] / cap_factor),
                    round(grid_cell.top_left[1] / cap_factor),
                ),
                patch_size=(coarse_width, coarse_height),
                fill_value=np.nan,
            ),
            nan_fill_value,
        )
        for grid_cell in grid_cells
    ]

    return Stage(
        image=pad_image_array(
            comparison_coarse, pad_width=coarse_width, pad_height=coarse_height
        ),
        templates=templates,
        fill_value=float(np.nanmean(comparison_coarse)),
    )


def compute_scale_match_factor(
    reference_image: ScanImage, comparison_image: ScanImage
) -> float:
    """
    Factor that puts *comparison_image* on the reference's pixel grid.

    1.0 when the two already share a pixel scale (including when *comparison_image* has already
    been resampled onto the reference's grid), so callers can pass either the original comparison
    image or an already-aligned one and get a consistent result.

    :param reference_image: Reference scan image.
    :param comparison_image: Comparison scan image, at any pixel scale.
    :returns: Multiplier such that ``comparison_image.scale_x * factor == reference_image.scale_x``.
    """
    return reference_image.scale_x / comparison_image.scale_x


def downsample(data: FloatArray2D, cap_factor: float) -> FloatArray2D:
    """
    Shrink image by *cap_factor* on both axes (NaN-aware area filter).

    :param data: Input 2D array.
    :param cap_factor: Pixels per output pixel (>= 1).
    :returns: Downsampled array.
    """
    return resample_nan_aware(data, factors=(cap_factor, cap_factor))


def record_results(
    grid_cells: list[GridCell],
    results: list[Match],
    padded_shape: tuple[int, ...],
    cell_width: int,
    cell_height: int,
) -> None:
    """
    Map each match from the padded canvas back onto its grid cell.

    :param grid_cells: Grid cells whose ``grid_search_params`` to update.
    :param results: One :class:`Match` per grid cell (in the same order).
    :param padded_shape: Shape of the padded comparison canvas used for matching.
    :param cell_width: Width of one grid cell in pixels.
    :param cell_height: Height of one grid cell in pixels.
    """
    for grid_cell, match in zip(grid_cells, results):
        center_x, center_y = map_canvas_to_image(
            match.x,
            match.y,
            (cell_height, cell_width),
            (padded_shape[0], padded_shape[1]),
            match.angle_deg,
        )
        grid_cell.grid_search_params.update(
            score=match.score,
            angle=match.angle_deg,
            center_x=center_x - cell_width,  # Undo the padding
            center_y=center_y - cell_height,
        )
