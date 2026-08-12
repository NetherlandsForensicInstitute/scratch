"""Cell registration: build the padded canvas and templates each search stage runs on."""

from __future__ import annotations

import numpy as np
from loguru import logger

from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.resample import (
    SCALE_MATCH_RTOL,
    resample_array_2d_nan_aware,
    select_interpolation,
)
from conversion.surface_comparison.cell_registration.geometry import pad_image_array
from conversion.surface_comparison.cell_registration.models import Stage
from conversion.surface_comparison.grid import extract_patch
from conversion.surface_comparison.models import ComparisonParams, GridCell
from conversion.surface_comparison.template_fill import fill_template_nan

#: Minimum coarse cell size for reliable matching. If downsampling would produce cells smaller
#: than this, the cap factor is reduced to keep coarse cells above this threshold.
_MIN_COARSE_CELL = 12


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
    :raises ValueError: If the two images are not on the same pixel scale. Comparing their pixel
        counts against *max_size* is only meaningful once a pixel means the same thing in both.
    """
    if not np.isclose(
        reference_image.scale_x, comparison_image.scale_x, rtol=SCALE_MATCH_RTOL
    ):
        raise ValueError(
            f"Reference ({reference_image.scale_x}) and comparison "
            f"({comparison_image.scale_x}) images must be on the same pixel scale; "
            "resample the comparison image first."
        )
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
    :raises ValueError: If the grid cells disagree on ``nan_fill_value``. The coarse stage picks
        each cell's candidate locations, so every template must be filled the same way the
        full-resolution ones were.
    """
    if len({grid_cell.nan_fill_value for grid_cell in grid_cells}) > 1:
        raise ValueError("All grid cells must share the same nan_fill_value.")
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
        data=resample_to_coarse(reference_image.data, cap_factor),
        scale_x=reference_image.scale_x * cap_factor,
        scale_y=reference_image.scale_y * cap_factor,
    )
    scale_match_factor = compute_scale_match_factor(reference_image, comparison_image)
    comparison_coarse = resample_to_coarse(
        comparison_image.data, cap_factor * scale_match_factor
    )

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


def resample_to_coarse(data: FloatArray2D, factor: float) -> FloatArray2D:
    """
    Put an image on the coarse grid, NaN-aware and in either direction.

    Usually a shrink, but not always: the comparison image reaches the coarse grid in one pass from
    its own native scale (see :func:`build_coarse_stage`), and a comparison scan coarser than the
    reference can need *factor* below 1.0. The interpolation therefore follows the direction rather
    than assuming ``area``, which cv2 degenerates to nearest-neighbor when zooming in.

    :param data: Input 2D array.
    :param factor: Source pixels per output pixel; above 1.0 shrinks, below 1.0 grows.
    :returns: Resampled array.
    """
    factors = (factor, factor)
    return resample_array_2d_nan_aware(data, factors, select_interpolation(factors))
