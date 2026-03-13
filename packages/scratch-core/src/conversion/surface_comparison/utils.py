from typing import Sequence

from container_models.base import FloatArray2D, FloatArray1D
from container_models.scan_image import ScanImage
from scipy.ndimage import rotate
import numpy as np

from conversion.surface_comparison.models import Cell


def convert_meters_to_pixels(
    values: tuple[float, float], pixel_size: float
) -> tuple[int, int]:
    """TODO: Remove this function if possible."""

    def _convert(value: float) -> int:
        return int(round(value / pixel_size))

    return _convert(values[0]), _convert(values[1])


def convert_pixels_to_meters(
    values: tuple[float, float], pixel_size: float
) -> tuple[float, float]:
    """TODO: Remove this function if possible."""

    def _convert(value: float) -> float:
        return value * pixel_size

    return _convert(values[0]), _convert(values[1])


def rotate_scan_image(scan_image: ScanImage, angle: float) -> ScanImage:
    """
    Rotate an instance of `ScanImage` by `angle` degrees.
    Background values are filled with NaNs.
    """
    return scan_image.model_copy(
        update={"data": rotate(scan_image.data, angle=angle, cval=np.nan)}
    )


def _cells_to_grid(
    cells: Sequence[Cell],
) -> tuple[FloatArray2D, FloatArray2D, FloatArray1D]:
    """
    Map unordered cells onto a row-major grid.

    Grid dimensions and spacing are inferred from the cell center positions.

    :param cells: Unordered cell results from the CMC pipeline.
    :return: cell_correlations (n_rows, n_cols),
             cell_positions_compared (n_rows * n_cols, 2) in µm,
             cell_rotations_compared (n_rows * n_cols,) in radians.
    """
    centers = np.array([c.center_reference for c in cells])

    unique_x = np.unique(np.round(centers[:, 0], decimals=9))
    unique_y = np.unique(np.round(centers[:, 1], decimals=9))
    step_x = np.diff(unique_x).min() if len(unique_x) > 1 else 1.0
    step_y = np.diff(unique_y).min() if len(unique_y) > 1 else 1.0

    col_indices = np.round((centers[:, 0] - unique_x[0]) / step_x).astype(int)
    row_indices = np.round((centers[:, 1] - unique_y[0]) / step_y).astype(int)

    n_rows = row_indices.max() + 1
    n_cols = col_indices.max() + 1

    cell_correlations = np.full((n_rows, n_cols), np.nan)
    cell_positions = np.full((n_rows * n_cols, 2), np.nan)
    cell_rotations = np.full(n_rows * n_cols, np.nan)

    for k, cell in enumerate(cells):
        r, c = row_indices[k], col_indices[k]
        flat = r * n_cols + c
        cell_correlations[r, c] = cell.best_score
        cell_positions[flat] = np.array(cell.center_comparison) * 1e6
        cell_rotations[flat] = np.deg2rad(cell.angle_deg)

    return cell_correlations, cell_positions, cell_rotations
