import numpy as np
import torch

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.coarse import coarse_to_fine_match
from conversion.surface_comparison.cell_registration.utils import (
    batched_match,
    canvas_to_image,
    convert_grid_cell_to_cell,
    pad_image_array,
)
from conversion.surface_comparison.models import (
    Cell,
    ComparisonParams,
    GridCell,
)


def match_cells(
    grid_cells: list[GridCell],
    comparison_image: ScanImage,
    params: ComparisonParams,
    device: "torch.device | None" = None,
    reduction: int | None = None,
    search_options: dict | None = None,
) -> list[Cell]:
    """
    Find the best-matching position and angle for each grid cell in the comparison image.

    For each angle in the configured sweep, the padded comparison image is rotated and a normalized
    cross-correlation score map is computed per cell. Positions whose comparison-patch fill fraction
    falls below ``params.minimum_fill_fraction`` are masked out. The rotation and translation that
    together yield the highest unmasked score are stored in each cell's :class:`GridSearchParams`.

    The comparison image is padded by a full cell in each direction before the search so that cells that
    lie near the image boundary can still be matched. After unrotating the cell center, the padding offset
    is subtracted back when the best position is recorded, so all stored coordinates are in the original
    (unpadded) pixel space.

    The search itself runs on GPU when one is available and on CPU otherwise, using the same code path;
    see :func:`batched_match`.

    :param grid_cells: Reference grid cells to register; all cells must have the same size.
    :param comparison_image: Comparison scan image to search over.
    :param params: Algorithm parameters (angle sweep bounds, step, fill-fraction threshold).
    :param device: Optional torch device override; defaults to CUDA when available. Mainly useful
        for benchmarking and for cross-device reproducibility checks.
    :param reduction: When set, locate cells with an exhaustive sweep at this reduction factor and
        then refine at full resolution, instead of searching everything at full resolution. The
        angle sweep stays global either way, so grossly misoriented marks are still found. Leave as
        ``None`` for the exhaustive search.
    :param search_options: Extra keyword arguments forwarded to the search, e.g.
        ``{"n_candidates": 2, "margin": 12}``. Ignored when *reduction* is ``None``.
    :returns: List of :class:`Cell` objects with the best registration result per grid cell.
    """
    if not grid_cells:
        return []

    fill_value_comparison = float(np.nanmean(comparison_image.data))
    pixel_size = comparison_image.scale_x  # Assumes isotropic image
    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    pad_width, pad_height = cell_width, cell_height  # Set pad size to cell size

    comparison_data = pad_image_array(
        comparison_image.data, pad_width=pad_width, pad_height=pad_height
    )
    angles = np.arange(
        params.search_angle_min,
        params.search_angle_max + params.search_angle_step,
        params.search_angle_step,
    )

    search = batched_match if reduction is None else coarse_to_fine_match
    extra = (
        {} if reduction is None else {"reduction": reduction, **(search_options or {})}
    )
    results = search(
        image=comparison_data,
        templates=[grid_cell.cell_data_filled for grid_cell in grid_cells],
        angles=angles,
        minimum_fill_fraction=params.minimum_fill_fraction,
        fill_value=fill_value_comparison,
        device=device,
        **extra,
    )

    for grid_cell, (score, x, y, angle) in zip(grid_cells, results):
        center_x, center_y = canvas_to_image(
            x, y, (cell_height, cell_width), comparison_data.shape, angle
        )
        grid_cell.grid_search_params.update(
            score=score,
            angle=angle,
            center_x=center_x - pad_width,  # Undo the padding
            center_y=center_y - pad_height,  # Undo the padding
        )

    return [
        convert_grid_cell_to_cell(grid_cell=grid_cell, pixel_size=pixel_size)
        for grid_cell in grid_cells
    ]
