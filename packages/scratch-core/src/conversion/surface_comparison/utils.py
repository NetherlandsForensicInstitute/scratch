from container_models.base import Points2D
from typing import Sequence

from container_models.base import FloatArray2D, FloatArray1D
import numpy as np

from conversion.surface_comparison.models import Cell


def convert_meters_to_pixels(
    values: tuple[float, float], pixel_size: float
) -> tuple[int, int]:
    """Convert x,y coordinates in meters into pixel coordinates."""

    def _convert(value: float) -> int:
        return int(round(value / pixel_size))

    return _convert(values[0]), _convert(values[1])


def convert_pixels_to_meters(
    values: tuple[float, float], pixel_size: float
) -> tuple[float, float]:
    """Convert x,y pixel coordinates into physical coordinates."""

    def _convert(value: float) -> float:
        return value * pixel_size

    return _convert(values[0]), _convert(values[1])


def rotate_points(
    points: Points2D, angle: float, center: tuple[float, float]
) -> Points2D:
    """
    Rotate 2-D points around a center.

    :param points: (N, 2) array of [x, y] coordinates.
    :param angle: Rotation angle in radians.
    :param center: Tuple for the center of rotation [x, y].
    :returns: (N, 2) rotated points.
    """
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
    translation = np.array(center)
    return (points - translation) @ rotation_matrix.T + translation


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
