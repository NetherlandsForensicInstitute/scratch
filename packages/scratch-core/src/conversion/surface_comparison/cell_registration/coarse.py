from container_models.base import BinaryMask, FloatArray2D, FloatArray1D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
    GridCell,
)
from collections.abc import Sequence
import numpy as np
from skimage.transform import rotate
from conversion.surface_comparison.cell_registration.utils import (
    convert_grid_cell_to_cell,
    pad_image_array,
)
from multiprocessing import Pool
from os import cpu_count
import cv2
from dataclasses import dataclass
from conversion.surface_comparison.utils import rotate_points

N_PROCESSES = cpu_count()


@dataclass(frozen=True)
class _WorkerTask:
    angles: FloatArray1D
    grid_cells: list[GridCell]
    comparison_data: FloatArray2D
    cell_size: tuple[int, int]
    minimum_fill_fraction: float
    fill_value: float
    padded_center: tuple[float, float]
    pad_size: tuple[int, int]


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
    pixel_size = comparison_image.scale_x  # Assumes isotropic image
    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    pad_width, pad_height = cell_width, cell_height  # Set pad size to cell size

    comparison_data = pad_image_array(
        comparison_image.data, pad_width=pad_width, pad_height=pad_height
    )
    padded_center = (
        (comparison_data.shape[1] - 1) / 2,
        (comparison_data.shape[0] - 1) / 2,
    )

    # Build the worker tasks for parallel processing
    angles = np.arange(
        params.search_angle_min, params.search_angle_max, params.search_angle_step
    )
    tasks = [
        _WorkerTask(
            angles=chunk,
            grid_cells=[cell.copy() for cell in grid_cells],
            comparison_data=comparison_data,
            cell_size=(cell_width, cell_height),
            minimum_fill_fraction=params.minimum_fill_fraction,
            fill_value=fill_value_comparison,
            padded_center=padded_center,
            pad_size=(pad_width, pad_height),
        )
        for chunk in np.array_split(angles, N_PROCESSES)  # type: ignore
    ]
    # Apply the map-reduce paradigm
    with Pool(N_PROCESSES) as pool:
        per_worker_results = pool.map(_run_iteration, tasks)
    best_cells = _reduce(per_worker_results)

    return [
        convert_grid_cell_to_cell(grid_cell=grid_cell, pixel_size=pixel_size)
        for grid_cell in best_cells
    ]


def _reduce(results: Sequence[Sequence[GridCell]]) -> list[GridCell]:
    sort_key = "top_left"
    reduced = {}
    for grid_cells in results:
        for grid_cell in grid_cells:
            key = getattr(grid_cell, sort_key)
            score = grid_cell.grid_search_params.score
            if key not in reduced or score > reduced[key].grid_search_params.score:
                reduced[key] = grid_cell
    return sorted(reduced.values(), key=lambda cell: getattr(cell, sort_key))


def _run_iteration(task: _WorkerTask) -> Sequence[GridCell]:
    cell_width, cell_height = task.cell_size
    pad_width, pad_height = task.pad_size
    for angle in task.angles:
        angle = float(angle)
        # Rotate the comparison image by `-angle` degrees.
        # This is equivalent to rotating the reference patch by `angle` degrees.
        rotated = rotate(
            image=task.comparison_data,
            angle=-angle,
            cval=np.nan,  # type: ignore
            order=0,
            resize=True,
        )
        # Get the mask of valid pixels for the rotated image
        valid_mask = ~np.isnan(rotated)
        # Compute the fill fraction mask based on the valid pixels mask
        fill_fraction_map = _get_fill_fraction_map(
            valid_pixel_mask=valid_mask,
            cell_width=cell_width,
            cell_height=cell_height,
        )
        fill_fraction_mask = fill_fraction_map >= task.minimum_fill_fraction
        # Now that we computed the fill fraction mask, we can safely replace NaN values in the rotated image
        rotated[~valid_mask] = task.fill_value

        for grid_cell in task.grid_cells:
            score_map = _get_score_map(
                comparison_array=rotated,
                template=grid_cell.cell_data_filled,
            )
            score, x, y = _compute_best_score_from_maps(
                score_map=score_map, fill_fraction_mask=fill_fraction_mask
            )
            if score > grid_cell.grid_search_params.score:
                # Compute the center coordinates of the cell on the (original) unrotated image
                cell_center = (x + cell_width / 2, y + cell_height / 2)
                rotated_center_x, rotated_center_y = (
                    (rotated.shape[1] - 1) / 2,  # type: ignore
                    (rotated.shape[0] - 1) / 2,
                )
                original_center_x, original_center_y = _unrotate_point(
                    rotated_point=cell_center,
                    original_image_center=task.padded_center,
                    rotated_image_center=(rotated_center_x, rotated_center_y),
                    angle_deg=angle,
                )
                # Update parameters
                grid_cell.grid_search_params.update(
                    score=score,
                    angle=angle,
                    center_x=original_center_x - pad_width,  # Undo the padding
                    center_y=original_center_y - pad_height,  # Undo the padding
                )
    return task.grid_cells


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
