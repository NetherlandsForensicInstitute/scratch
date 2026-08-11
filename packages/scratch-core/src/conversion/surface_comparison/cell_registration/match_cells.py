import numpy as np
import torch

from container_models.scan_image import ScanImage
from conversion.resample import resample_array_2d
from conversion.surface_comparison.cell_registration.coarse import coarse_to_fine_match
from conversion.surface_comparison.cell_registration.utils import (
    canvas_to_image,
    convert_grid_cell_to_cell,
    pad_image_array,
)
from conversion.surface_comparison.grid import extract_patch
from conversion.surface_comparison.models import (
    Cell,
    ComparisonParams,
    GridCell,
)

from loguru import logger

from conversion.surface_comparison.template_fill import fill_template_nan

#: Minimum coarse cell size for reliable matching. If downsampling would produce cells smaller
#: than this, the cap factor is reduced to keep coarse cells above this threshold.
_MIN_COARSE_CELL = 12


def match_cells(
    grid_cells: list[GridCell],
    reference_image: ScanImage,
    comparison_image: ScanImage,
    params: ComparisonParams,
    device: "torch.device | None" = None,
) -> list[Cell]:
    """
    Find the best-matching position and angle for each grid cell in the comparison image.

    Two-stage search, both stages sharing the same underlying scoring code (see
    :mod:`cell_registration.utils` and :func:`cell_registration.coarse.coarse_to_fine_match`):

    1. **Coarse**: the reference and comparison images are downsampled together, once, to a shared
       pixel scale capped at ``params.max_size`` pixels on the longer side, and an exhaustive
       translation + rotation sweep runs on that pair, keeping ``params.n_candidates`` candidate
       poses per cell.
    2. **Fine**: each candidate is refined at full resolution, searching ``params.fine_n_pixels``
       pixels of translation and ``params.fine_m_degrees`` degrees of rotation (1-degree steps)
       around that candidate's own position and angle.

    *reference_image* and *comparison_image* may be at different native pixel scales; both are
    resampled to a common scale from their original data in a single interpolation pass each -
    deliberately not by resampling once to align scale and again to cap size, which would chain two
    lossy resizes to reach one target.

    Coarse-stage templates are *not* obtained by downsampling each cell's patch in isolation.
    Downsampling the whole reference image once and then extracting each coarse cell from it (via
    the same :func:`~conversion.surface_comparison.grid.extract_patch` used to build the full
    resolution grid) gives every template's edge pixels the same neighbouring context the
    comparison canvas gets; downsampling per-cell would compute those edges from cell-local data
    only, which the comparison canvas' edges never do, silently penalising the true match.

    NaNs are handled explicitly throughout. The comparison images keep NaN outside real data: the
    search consumes it directly to compute a per-position fill fraction and reject sparse windows.
    Reference templates fill NaN according to ``params.template_nan_fill_strategy``, resolved once
    in the pipeline and carried on each cell as ``GridCell.nan_fill_value`` - by default each
    cell's own valid-pixel mean, so that after template centering a missing pixel is exactly zero
    and drops out of the correlation entirely instead of behaving like real, flat surface data.
    Coarse and full-resolution templates are filled with the *same* resolved value, both via
    :func:`~conversion.surface_comparison.template_fill.fill_template_nan`: the coarse stage is what
    chooses each cell's candidate locations, so filling it differently would change where a cell
    matches, not merely how that match is refined.

    :param grid_cells: Reference grid cells to register; all cells must have the same size and the
        same ``nan_fill_value``.
    :param reference_image: Reference scan image the grid cells were generated from.
    :param comparison_image: Comparison scan image to search over, at its own native pixel scale.
    :param params: Algorithm parameters (angle sweep, coarse/fine search configuration).
    :param device: Optional torch device override; defaults to CUDA when available. Mainly useful
        for benchmarking and for cross-device reproducibility checks.
    :returns: List of :class:`Cell` objects with the best registration result per grid cell.
    :raises ValueError: If the grid cells do not all share the same ``nan_fill_value``.
    """
    if not grid_cells:
        return []
    if len({grid_cell.nan_fill_value for grid_cell in grid_cells}) > 1:
        raise ValueError("All grid cells must share the same nan_fill_value.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pixel_size = (
        reference_image.scale_x
    )  # Assumes isotropic image; the shared output scale.
    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height

    # --- Bring the comparison image to the reference's pixel scale, in one pass. ---
    scale_match_factor = pixel_size / comparison_image.scale_x
    if not np.isclose(scale_match_factor, 1.0, rtol=1e-5):
        logger.info(f"Resampling comparison image with factor {scale_match_factor}")
        comparison_full_data = resample_array_2d(
            comparison_image.data,
            factors=(scale_match_factor, scale_match_factor),
            interpolation=params.resample_interpolation,
        )
    else:
        # Scales already agree to within a rounding error; resampling here would be a lossy
        # interpolation pass that changes nothing.
        logger.info("Skipping resampling of images. Scales already match.")
        comparison_full_data = comparison_image.data

    # --- Cap the shared canvas to params.max_size for the coarse stage. ---
    largest_dimension = max(
        reference_image.height,
        reference_image.width,
        comparison_full_data.shape[0],
        comparison_full_data.shape[1],
    )

    # Compute the raw downsampling multiplier derived from the maximum size limit
    raw_cap_factor = max(1.0, largest_dimension / params.max_size)

    # Apply a lower bound based on cell size: coarse cells must not become too small.
    # If cap_factor is too large, coarse cells shrink and matching becomes unreliable.
    cell_min_dim = min(cell_width, cell_height)
    min_allowed_cap = max(1.0, cell_min_dim / _MIN_COARSE_CELL)
    cap_factor = min(raw_cap_factor, min_allowed_cap)

    if raw_cap_factor > min_allowed_cap:
        logger.info(
            "Reduced cap factor from {:.2f} to {:.2f}: cells are {}x{} px, "
            "and a coarse cell below {} px cannot localise reliably.",
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
        params.max_size,
        raw_cap_factor,
        cap_factor,
        cap_factor > 1.0,
    )

    fill_value_full = float(np.nanmean(comparison_full_data))
    comparison_full_padded = pad_image_array(
        comparison_full_data, pad_width=cell_width, pad_height=cell_height
    )
    templates_full = [grid_cell.cell_data_filled for grid_cell in grid_cells]

    if cap_factor > 1.0:
        # Single pass directly from the raw comparison data, combining scale-match and size-cap
        # into one interpolation instead of shrinking the already-resampled comparison_full again.
        combined_factor = scale_match_factor * cap_factor
        comparison_coarse_data = resample_array_2d(
            comparison_image.data,
            factors=(combined_factor, combined_factor),
            interpolation=params.resample_interpolation,
        )
        reference_coarse_data = resample_array_2d(
            reference_image.data,
            factors=(cap_factor, cap_factor),
            interpolation=params.resample_interpolation,
        )
        reference_coarse = ScanImage(
            data=reference_coarse_data,
            scale_x=reference_image.scale_x * cap_factor,
            scale_y=reference_image.scale_y * cap_factor,
        )
        coarse_cell_width = max(1, int(np.ceil(cell_width / cap_factor)))
        coarse_cell_height = max(1, int(np.ceil(cell_height / cap_factor)))

        logger.info(
            "Coarse stage: {}x{} px cells -> {}x{} px coarse cells (cap={:.2f})",
            cell_width,
            cell_height,
            coarse_cell_width,
            coarse_cell_height,
            cap_factor,
        )

        # All cells carry the same resolved fill value (generate_grid applies one to the whole grid).
        # Coarse templates must use it too: the coarse stage picks the candidate locations, so
        # filling it differently changes where each cell matches, not just how it is polished.
        coarse_nan_fill = grid_cells[0].nan_fill_value

        templates_coarse = []
        for grid_cell in grid_cells:
            coarse_top_left = (
                round(grid_cell.top_left[0] / cap_factor),
                round(grid_cell.top_left[1] / cap_factor),
            )
            patch = extract_patch(
                scan_image=reference_coarse,
                coordinates=coarse_top_left,
                patch_size=(coarse_cell_width, coarse_cell_height),
                fill_value=np.nan,
            )
            templates_coarse.append(fill_template_nan(patch, coarse_nan_fill))

        fill_value_coarse = float(np.nanmean(comparison_coarse_data))
        comparison_coarse_padded = pad_image_array(
            comparison_coarse_data,
            pad_width=coarse_cell_width,
            pad_height=coarse_cell_height,
        )
    else:
        # Images already fit within max_size: coarse and fine would search the same resolution, so
        # reuse the full-resolution arrays and let coarse_to_fine_match take its single-pass shortcut.
        templates_coarse = templates_full
        comparison_coarse_padded = comparison_full_padded
        fill_value_coarse = fill_value_full
        logger.info(
            "Coarse stage skipped: images fit within max_size ({} px), "
            "matching at full resolution ({} x {} px)",
            params.max_size,
            reference_image.width,
            reference_image.height,
        )

    angles = np.arange(
        params.search_angle_min,
        params.search_angle_max + params.search_angle_step,
        params.search_angle_step,
    )

    results = coarse_to_fine_match(
        image_full=comparison_full_padded,
        image_coarse=comparison_coarse_padded,
        templates_full=templates_full,
        templates_coarse=templates_coarse,
        cap_factor=cap_factor,
        angles=angles,
        minimum_fill_fraction=params.minimum_fill_fraction,
        fill_value_full=fill_value_full,
        fill_value_coarse=fill_value_coarse,
        n_candidates=params.n_candidates,
        position_margin=params.fine_n_pixels,
        angle_margin_degrees=params.fine_m_degrees,
        template_batch_size=params.template_batch_size,
        angle_batch_size=params.angle_batch_size,
        fine_batch_size=params.fine_batch_size,
        device=device,
    )

    for grid_cell, (score, x, y, angle) in zip(grid_cells, results):
        center_x, center_y = canvas_to_image(
            x, y, (cell_height, cell_width), comparison_full_padded.shape, angle
        )
        grid_cell.grid_search_params.update(
            score=score,
            angle=angle,
            center_x=center_x - cell_width,  # Undo the padding
            center_y=center_y - cell_height,  # Undo the padding
        )

    return [
        convert_grid_cell_to_cell(grid_cell=grid_cell, pixel_size=pixel_size)
        for grid_cell in grid_cells
    ]
