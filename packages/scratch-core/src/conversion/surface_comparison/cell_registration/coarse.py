import numpy as np

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.utils import (
    convert_grid_cell_to_cell,
    pad_image_array,
    _batched_match,
    _rotate_image,
)
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
    GridCell,
)
from conversion.surface_comparison.utils import rotate_points


def match_cells(
    grid_cells: list[GridCell], comparison_image: ScanImage, params: ComparisonParams
) -> list[Cell]:
    """
    Find the best-matching position and angle for each grid cell in the comparison image.

    For each angle in the configured sweep, the padded comparison image is rotated and a normalized
    cross-correlation score map is computed per cell using ``cv2.TM_CCOEFF_NORMED``. Positions whose
    comparison-patch fill fraction falls below ``params.minimum_fill_fraction`` are masked out.
    Per rotation angle, the highest score with its corresponding translation is stored.
    The rotation that yields the highest unmasked score will be stored in each cell's :class:`GridSearchParams`.

    The comparison image is padded by a full cell in each direction before the search so that cells that
    lie near the image boundary can still be matched. After unrotating the cell center, the padding offset
    is subtracted back when the best position is recorded, so all stored coordinates are in the original
    (unpadded) pixel space.

    :param grid_cells: Reference grid cells to register; all cells must have the same size.
    :param comparison_image: Comparison scan image to search over.
    :param params: Algorithm parameters (angle sweep bounds, step, fill-fraction threshold).
    :returns: List of :class:`Cell` objects with the best registration result per grid cell.
    """
    if not grid_cells:
        return []

    fill_value_comparison = float(np.nanmean(comparison_image.data))
    pixel_size = comparison_image.scale_x
    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    pad_width, pad_height = cell_width, cell_height

    comparison_data = pad_image_array(
        comparison_image.data, pad_width=pad_width, pad_height=pad_height
    )
    padded_center = (
        (comparison_data.shape[1] - 1) / 2,
        (comparison_data.shape[0] - 1) / 2,
    )

    angles = np.arange(
        params.search_angle_min,
        params.search_angle_max + params.search_angle_step,
        params.search_angle_step,
    )

    templates = [gc.cell_data_filled for gc in grid_cells]
    results = _batched_match(
        comparison_data,
        templates,
        angles,
        params.minimum_fill_fraction,
        fill_value_comparison,
    )
    for grid_cell, (score, x, y, angle_idx) in zip(grid_cells, results):
        angle = float(angles[angle_idx])
        rotated = _rotate_image(comparison_data, angle, fill_value=np.nan)
        rot_h, rot_w = rotated.shape

        cell_center = (x + cell_width / 2, y + cell_height / 2)
        rotated_center = ((rot_w - 1) / 2, (rot_h - 1) / 2)

        orig_x, orig_y = _unrotate_point(
            rotated_point=cell_center,
            original_image_center=padded_center,
            rotated_image_center=rotated_center,
            angle_deg=angle,
        )
        grid_cell.grid_search_params.update(
            score=score,
            angle=angle,
            center_x=orig_x - pad_width,
            center_y=orig_y - pad_height,
        )

    return [
        convert_grid_cell_to_cell(grid_cell=grid_cell, pixel_size=pixel_size)
        for grid_cell in grid_cells
    ]


def _unrotate_point(
    rotated_point: tuple[float, float],
    original_image_center: tuple[float, float],
    rotated_image_center: tuple[float, float],
    angle_deg: float,
) -> tuple[float, float]:
    """
    Map a match coordinate from the rotated output back to the original comparison image.

    :param rotated_point: The (x, y) coordinates of the point in the rotated image.
    :param original_image_center: The center (x, y) coordinates of the original unrotated image.
    :param rotated_image_center: The center (x, y) coordinates  of the rotated image.
    :param angle_deg: The rotation angle in degrees for the rotated point.
    """
    x_center, y_center = original_image_center
    x_center_rotated, y_center_rotated = rotated_image_center
    x_rotated, y_rotated = rotated_point
    # Shift the coordinate relative to the center of the rotated image
    dx, dy = x_rotated - x_center_rotated, y_rotated - y_center_rotated
    # Unrotate vector
    x, y = rotate_points(
        points=np.array([(dx, dy)]), center=(0, 0), angle=-np.radians(angle_deg)
    )[0]
    # Shift the coordinates relative to the top-left of the original image.
    return x_center + x, y_center + y
