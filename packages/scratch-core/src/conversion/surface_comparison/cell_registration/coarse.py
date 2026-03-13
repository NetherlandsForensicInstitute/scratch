from container_models.base import BinaryMask, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.grid import GridCell
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
)
import numpy as np
from skimage.transform import rotate
from conversion.surface_comparison.cell_registration.utils import (
    convert_grid_cell_to_cell,
    pad_image_array,
)

import cv2

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

    The comparison image is padded by half a cell in each direction before the search so that cells whose
    reference top-left lies near the image boundary can still be matched. The padding offset is subtracted
    back when the best position is recorded, so all stored coordinates are in the original (unpadded) pixel space.

    :param grid_cells: Reference grid cells to register; all cells must have the same size.
    :param comparison_image: Comparison scan image to search over.
    :param params: Algorithm parameters (angle sweep bounds, step, fill-fraction threshold).
    :returns: List of :class:`Cell` objects with the best registration result per grid cell.
    """
    if not grid_cells:
        return []

    fill_value_comparison = float(np.nanmean(comparison_image.data))
    pixel_size = comparison_image.scale_x  # Assumes isotropic image
    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    comparison_data = pad_image_array(
        comparison_image.data, pad_width=cell_width, pad_height=cell_height
    )
    angles = np.arange(
        params.search_angle_min, params.search_angle_max, params.search_angle_step
    )

    for angle in angles:
        angle = float(angle)
        # Rotate the comparison image by `-angle` degrees.
        # This is equivalent to rotating the reference patch by `angle` degrees.
        rotated = rotate(
            image=comparison_data,
            angle=-angle,
            cval=np.nan,  # type: ignore
            order=0,
            resize=False,
        )
        # Get the mask of valid pixels for the rotated image
        valid_mask = ~np.isnan(rotated)
        # Compute the fill fraction mask based on the valid pixels mask
        fill_fraction_map = _get_fill_fraction_map(
            valid_pixel_mask=valid_mask,
            cell_width=cell_width,
            cell_height=cell_height,
        )
        fill_fraction_mask = fill_fraction_map >= params.minimum_fill_fraction
        # Now that we computed the fill fraction mask, we can safely replace NaN values in the rotated image
        rotated[~valid_mask] = fill_value_comparison
        global_center_x, global_center_y = rotated.shape[1] / 2, rotated.shape[0] / 2  # type: ignore

        for grid_cell in grid_cells:
            score_map = _get_score_map(
                comparison_array=rotated,
                template=grid_cell.cell_data_filled,
            )
            score, x, y = _compute_best_score_from_maps(
                score_map=score_map, fill_fraction_mask=fill_fraction_mask
            )
            if score > grid_cell.grid_search_params.score:
                # Compute the center coordinates of the cell on the (original) unrotated image
                center_x, center_y = _compute_unrotated_cell_center(
                    rotated_top_left=(x, y),
                    angle=angle,
                    pad_size=(cell_width, cell_height),
                    center=(global_center_x, global_center_y),
                )
                grid_cell.grid_search_params.update(
                    score=score,
                    angle=angle,
                    center_x=center_x,
                    center_y=center_y,
                )

    return [
        convert_grid_cell_to_cell(grid_cell=grid_cell, pixel_size=pixel_size)
        for grid_cell in grid_cells
    ]


def _get_fill_fraction_map(
    valid_pixel_mask: BinaryMask,
    cell_height: int,
    cell_width: int,
) -> FloatArray2D:
    """
    Compute a 2D map where each entry [y, x] is the fill fraction of a cell-sized window with its
    **top-left corner** at pixel (x, y), matching the indexing convention of ``cv2.matchTemplate``.

    :param valid_pixel_mask: Boolean array (H, W); True where image data is valid.
    :param cell_height: Height of the cell window in pixels.
    :param cell_width: Width of the cell window in pixels.
    :returns: Float64 array (H, W) with fill fractions in [0, 1], top-left indexed.
        Entries near the bottom-right boundary are underestimates and will be rejected by the fill-fraction gate.
        Since the image is padded with NaNs before calling this function, this does not matter.
    """
    kernel = np.ones((cell_height, cell_width), dtype=np.float32) / (
        cell_height * cell_width
    )
    filtered = cv2.filter2D(
        valid_pixel_mask.astype(np.float32),
        ddepth=-1,
        kernel=kernel,
        anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.asarray(filtered, dtype=np.float64)


def _compute_unrotated_cell_center(
    rotated_top_left: tuple[float, float],
    angle: float,
    pad_size: tuple[int, int],
    center: tuple[float, float],
):
    # Undo the -angle rotation that was applied to the padded comparison image.
    angle_rad = np.radians(-angle)
    pad_x, pad_y = pad_size
    x_original, y_original = rotate_points(
        points=np.array([rotated_top_left]), center=center, angle=angle_rad
    )[0]
    # Remove the padding (one full cell on each side)
    top_left_x = x_original - pad_x
    top_left_y = y_original - pad_y
    # Compute the original center coordinates from the top-left coordinates and the angle
    center_x, center_y = rotate_points(
        points=np.array([(pad_x / 2, pad_y / 2)]), angle=angle_rad, center=(0, 0)
    )[0]
    center_x += top_left_x
    center_y += top_left_y
    return center_x, center_y


def _compute_best_score_from_maps(
    score_map: FloatArray2D, fill_fraction_mask: BinaryMask
) -> tuple[float, int, int]:
    """
    Compute the highest correlation score and the corresponding x, y coordinates
    from the score and fill fraction maps.
    """
    # Make sure the shape of `score_map` and the `fill_fraction_mask` match, and
    # discard irrelevant fill fraction mask positions at the bottom right.
    valid_positions = fill_fraction_mask[: score_map.shape[0], : score_map.shape[1]]
    # Replace non-valid values (where fill fraction is below threshold) with -inf
    masked_scores = np.where(valid_positions, score_map, -np.inf)
    # Compute the best score and x, y position from the score map
    best_flat_index = np.argmax(masked_scores)
    score = masked_scores.flat[best_flat_index]
    y, x = np.unravel_index(best_flat_index, masked_scores.shape)
    return float(score), int(x), int(y)


def _get_score_map(
    comparison_array: FloatArray2D, template: FloatArray2D
) -> FloatArray2D:
    """
    Compute a normalized cross-correlation score map for one reference cell.

    Slides the cell template over the comparison array using ``cv2.TM_CCOEFF_NORMED``, which computes
    the Pearson correlation coefficient between the template and every same-sized patch. NaN values must
    have been replaced in both arrays before calling this function.

    :param comparison_array: NaN-free float32-compatible comparison image, padded by a full cell on each side.
    :param template: Reference grid cell whose ``cell_data`` is used as the template; must contain no NaN values.
    :returns: Float64 score map of shape ``(H - cell_height + 1, W - cell_width + 1)`` with values in ``[-1, 1]``.
    """

    score_map = cv2.matchTemplate(
        image=comparison_array.astype(np.float32),
        templ=template.astype(np.float32),
        method=cv2.TM_CCOEFF_NORMED,
    )
    return np.asarray(score_map, dtype=np.float64)
