from container_models.base import FloatArray2D
from conversion.surface_comparison.models import Cell, GridCell, CellMetaData

import numpy as np
from numpy.typing import NDArray

from conversion.surface_comparison.utils import convert_pixels_to_meters


def convert_grid_cell_to_cell(grid_cell: GridCell, pixel_size: float) -> Cell:
    """Convert an instance of `GridCell` to an instance of `Cell`."""
    cell = Cell(
        center_reference=convert_pixels_to_meters(
            values=grid_cell.center, pixel_size=pixel_size
        ),
        cell_data=grid_cell.cell_data,
        # TODO: add cell size in meters here
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
    """
    Pad a 2D array symmetrically with a constant fill value.

    Adds ``pad_height`` rows above and below and ``pad_width`` columns to the left and right of the input array.
    The original data is placed in the center of the output; the border is filled with ``fill_value``.

    :param array: Input 2D array of shape ``(height, width)``.
    :param pad_width: Number of columns to add on each side (left and right).
    :param pad_height: Number of rows to add on each side (top and bottom).
    :param fill_value: Constant value written into the padded border; defaults to NaN.
    :returns: Padded array of shape ``(height + 2 * pad_height, width + 2 * pad_width)``, same dtype as input.
    """
    height, width = array.shape
    new_shape = height + 2 * pad_height, width + 2 * pad_width
    output = np.full(shape=new_shape, fill_value=fill_value, dtype=array.dtype)
    output[pad_height : pad_height + height, pad_width : pad_width + width] = array
    return output
