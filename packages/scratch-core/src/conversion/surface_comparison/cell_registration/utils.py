from container_models.base import FloatArray2D
from conversion.surface_comparison.models import Cell, GridCell, CellMetaData

import numpy as np
from numpy.typing import NDArray

from conversion.surface_comparison.utils import convert_pixels_to_meters


def convert_grid_cell_to_cell(grid_cell: GridCell, pixel_size: float) -> Cell:
    cell = Cell(
        center_reference=convert_pixels_to_meters(
            values=grid_cell.center, pixel_size=pixel_size
        ),
        cell_data=grid_cell.cell_data,
        fill_fraction_reference=grid_cell.fill_fraction,
        best_score=grid_cell.grid_search_params.score,
        angle_deg=grid_cell.grid_search_params.angle,
        center_comparison=convert_pixels_to_meters(
            values=(
                grid_cell.grid_search_params.top_left_x + grid_cell.width / 2,
                grid_cell.grid_search_params.top_left_y + grid_cell.height / 2,
            ),
            pixel_size=pixel_size,
        ),
        is_congruent=False,  # TODO: We shouldn't set this here
        meta_data=CellMetaData(
            is_outlier=False, residual_angle_deg=0.0, position_error=(0, 0)
        ),  # TODO: We shouldn't set this here
    )
    return cell


def compute_fill_fraction(array: NDArray) -> float:
    """Compute the fraction of valid (non-NaN) values in the array."""
    return float(np.count_nonzero(~np.isnan(array)) / array.size)


def pad_image_array(
    array: FloatArray2D, pad_width: int, pad_height: int, fill_value: float = np.nan
) -> FloatArray2D:
    """TODO: Write docstring."""
    rows, cols = array.shape
    new_shape = rows + 2 * pad_height, cols + 2 * pad_width
    output = np.full(shape=new_shape, fill_value=fill_value, dtype=array.dtype)
    output[pad_height : pad_height + rows, pad_width : pad_width + cols] = array
    return output
