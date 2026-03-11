from collections.abc import Iterable

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


def match_cells(
    grid_cells: Iterable[GridCell],
    comparison_image: ScanImage,
    params: ComparisonParams,
    fill_value_reference: float,
) -> list[Cell]:
    """TODO: Write docstring."""
    grid_cells = list(grid_cells)
    if not grid_cells:
        return []

    angle = params.search_angle_min
    fill_value_comparison = float(np.nanmean(comparison_image.data))
    pixel_size = comparison_image.scale_x  # Assumes isotropic image

    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    pad_width, pad_height = cell_width // 2, cell_height // 2
    comparison_data = pad_image_array(
        comparison_image.data, pad_width=pad_width, pad_height=pad_height
    )
    templates = _build_templates(grid_cells=grid_cells, fill_value=fill_value_reference)

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
        # TODO: Do we need a different fill fraction for comparison?
        fill_fraction_mask = fill_fraction_map >= params.minimum_fill_fraction
        # Now that we computed the fill fraction mask, we can safely replace NaN values in the rotated image
        rotated[~valid_mask] = fill_value_comparison

        for grid_cell, template in zip(grid_cells, templates):
            score_map = _get_score_map(
                comparison_array_filled=rotated,
                cell=template,
            )
            # Make sure the shape of `score_map` and the `fill_fraction_mask` match
            sliced_mask = fill_fraction_mask[: score_map.shape[0], : score_map.shape[1]]
            # Replace non-valid values (where fill fraction is below threshold) with -inf
            masked_scores = np.where(sliced_mask, score_map, -np.inf)
            # Compute the best x, y position from the score map
            best_flat_index = int(np.argmax(masked_scores))
            score = float(masked_scores.flat[best_flat_index])
            if score > grid_cell.grid_search_params.score:
                y, x = np.unravel_index(best_flat_index, masked_scores.shape)
                grid_cell.grid_search_params.update(
                    score=score,
                    angle=angle,
                    top_left_x=int(x) - pad_width,
                    top_left_y=int(y) - pad_height,
                )

    output = [
        convert_grid_cell_to_cell(grid_cell=cell, pixel_size=pixel_size)
        for cell in grid_cells
    ]

    return output


def _get_fill_fraction_map(
    valid_pixel_mask: BinaryMask,
    cell_height: int,
    cell_width: int,
) -> FloatArray2D:
    """
    Compute a 2D map where each entry [y, x] is the fill fraction of a
    cell-sized window with its **top-left corner** at pixel (x, y), matching
    the indexing convention of ``cv2.matchTemplate``.

    :param valid_pixel_mask: Boolean array (H, W); True where image data is valid.
    :param cell_height: Height of the cell window in pixels.
    :param cell_width: Width of the cell window in pixels.
    :returns: Float64 array (H, W) with fill fractions in [0, 1], top-left indexed.
              Entries near the bottom-right boundary are underestimates and will
              be rejected by the fill-fraction gate.
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


def _get_score_map(
    comparison_array_filled: FloatArray2D, cell: GridCell
) -> FloatArray2D:
    """TODO: Write docstring."""
    score_map = cv2.matchTemplate(
        image=comparison_array_filled.astype(np.float32),
        templ=cell.cell_data.astype(np.float32),
        method=cv2.TM_CCOEFF_NORMED,
    )
    return np.asarray(score_map, dtype=np.float64)


def _build_templates(grid_cells: list[GridCell], fill_value: float) -> list[GridCell]:
    """TODO: Write docstring."""
    templates = []
    for grid_cell in grid_cells:
        # Deep copy cell and fill NaN values with numerical value
        cell_copy = grid_cell.copy()
        cell_copy.fill_nans(fill_value=fill_value)
        templates.append(cell_copy)
    return templates
