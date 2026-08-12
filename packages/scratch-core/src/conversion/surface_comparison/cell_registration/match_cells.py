"""Entry point of cell registration: prepare both resolution levels, search, report per-cell results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.resample import resample_array_2d
from conversion.surface_comparison.cell_registration.coarse_to_fine import (
    match_coarse_to_fine,
)
from conversion.surface_comparison.cell_registration.geometry import (
    map_canvas_to_image,
    pad_image_array,
)
from conversion.surface_comparison.cell_registration.models import Match
from conversion.surface_comparison.grid import extract_patch
from conversion.surface_comparison.models import (
    RESAMPLE_METHOD,
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
class _Stage:
    """One resolution level's padded comparison canvas and the templates to search it with."""

    image: FloatArray2D
    templates: list[FloatArray2D]
    fill_value: float


def match_cells(
    grid_cells: list[GridCell],
    reference_image: ScanImage,
    comparison_image: ScanImage,
    params: ComparisonParams,
    device: torch.device | None = None,
) -> list[Cell]:
    """
    Find the best-matching position and angle for each grid cell in the comparison image.

    Two stages, both sharing the same scoring code (see :mod:`.scoring`):

    1. **Coarse**: both images are downsampled until the larger fits within ``params.max_size``
       pixels, and an exhaustive translation + rotation sweep runs on that pair, keeping
       ``params.n_candidates`` candidate poses per cell.
    2. **Fine**: each candidate is refined at full resolution, searching ``params.fine_n_pixels``
       pixels of translation and ``params.fine_m_degrees`` degrees of rotation around it.

    Both images must already be on the same pixel scale; the pipeline resamples the comparison
    image before calling this.

    NaNs are handled explicitly throughout. The comparison image keeps NaN outside real data: the
    search consumes it to compute a per-position fill fraction and reject sparse windows. Reference
    templates fill NaN with ``GridCell.nan_fill_value``, resolved once in the pipeline.

    :param grid_cells: Reference grid cells to register; all must share size and ``nan_fill_value``.
    :param reference_image: Reference scan image the grid cells were generated from.
    :param comparison_image: Comparison scan image to search over, at the reference's pixel scale.
    :param params: Algorithm parameters (angle sweep, coarse/fine search configuration).
    :param device: Optional torch device override; defaults to CUDA when available.
    :returns: One :class:`Cell` per grid cell, with its best registration result.
    :raises ValueError: If the grid cells disagree on ``nan_fill_value``, or the two images are not
        on the same pixel scale.
    """
    if not grid_cells:
        return []
    if len({grid_cell.nan_fill_value for grid_cell in grid_cells}) > 1:
        raise ValueError("All grid cells must share the same nan_fill_value.")
    if not np.isclose(reference_image.scale_x, comparison_image.scale_x, rtol=1e-5):
        raise ValueError(
            f"Reference ({reference_image.scale_x}) and comparison "
            f"({comparison_image.scale_x}) images must be on the same pixel scale; "
            "resample the comparison image first."
        )

    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    cap_factor = compute_cap_factor(
        reference_image, comparison_image, cell_width, cell_height, params.max_size
    )
    full = build_full_resolution_stage(
        comparison_image, grid_cells, cell_width, cell_height
    )
    coarse = (
        build_coarse_stage(
            comparison_image,
            reference_image,
            grid_cells,
            cap_factor,
            cell_width,
            cell_height,
        )
        if cap_factor > 1.0
        else full
    )

    results = match_coarse_to_fine(
        image_full=full.image,
        image_coarse=coarse.image,
        templates_full=full.templates,
        templates_coarse=coarse.templates,
        cap_factor=cap_factor,
        angles=build_angle_sweep(params),
        minimum_fill_fraction=params.minimum_fill_fraction,
        fill_value_full=full.fill_value,
        fill_value_coarse=coarse.fill_value,
        n_candidates=params.n_candidates,
        position_margin=params.fine_n_pixels,
        angle_margin_degrees=params.fine_m_degrees,
        template_batch_size=params.template_batch_size,
        angle_batch_size=params.angle_batch_size,
        fine_batch_size=params.fine_batch_size,
        device=device,
    )

    record_results(grid_cells, results, full.image.shape, cell_width, cell_height)
    return [
        convert_grid_cell_to_cell(
            grid_cell=grid_cell, pixel_size=reference_image.scale_x
        )
        for grid_cell in grid_cells
    ]


def convert_grid_cell_to_cell(grid_cell: GridCell, pixel_size: float) -> Cell:
    """Convert an instance of `GridCell` to an instance of `Cell`."""
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
    """The coarse stage's angle sweep, in degrees, inclusive of both bounds."""
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
    How many full-resolution pixels one coarse pixel should span.

    Enough to bring the larger image within *max_size*, but never so much that coarse cells shrink
    below :data:`_MIN_COARSE_CELL` pixels, where they can no longer localize reliably.
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
) -> _Stage:
    """The comparison canvas and templates as they come out of the pipeline, only padded."""
    return _Stage(
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
    cell_width: int,
    cell_height: int,
) -> _Stage:
    """
    Both images downsampled by *cap_factor*, with templates re-cut from the coarse reference.

    Coarse templates are *not* obtained by downsampling each cell's patch in isolation.
    Downsampling the whole reference image once and re-extracting each cell from it gives every
    template's edge pixels the same neighboring context the comparison canvas gets; per-cell
    downsampling would compute those edges from cell-local data only, silently penalizing the true
    match.
    """
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
    comparison_coarse = downsample(comparison_image.data, cap_factor)

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

    return _Stage(
        image=pad_image_array(
            comparison_coarse, pad_width=coarse_width, pad_height=coarse_height
        ),
        templates=templates,
        fill_value=float(np.nanmean(comparison_coarse)),
    )


def downsample(data: FloatArray2D, cap_factor: float) -> FloatArray2D:
    """Shrink an image by *cap_factor* on both axes, using this pipeline's resampling method."""
    return resample_array_2d(
        data, factors=(cap_factor, cap_factor), method=RESAMPLE_METHOD
    )


def record_results(
    grid_cells: list[GridCell],
    results: list[Match],
    padded_shape: tuple[int, ...],
    cell_width: int,
    cell_height: int,
) -> None:
    """Map each match back off the rotated canvas and out of the padding, onto its grid cell."""
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
