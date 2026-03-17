from container_models.base import Points2D
from typing import Sequence

from container_models.base import FloatArray2D
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


def _cells_correlation_to_grid(cells: Sequence[Cell]) -> FloatArray2D:
    """
    Map unordered cells onto a row-major grid with the correlation as values.

    Grid dimensions and spacing are inferred from the cell center positions.

    :param cells: Unordered cell results from the CMC pipeline.
    :return: cell_correlations (n_rows, n_cols),
    """
    centers = np.array([c.center_reference for c in cells])

    unique_x = np.unique(np.round(centers[:, 0], decimals=9))
    unique_y = np.unique(np.round(centers[:, 1], decimals=9))
    min_x = unique_x[0]
    min_y = unique_y[0]
    max_x = unique_x[-1]
    max_y = unique_y[-1]
    step_x = (max_x - min_x) / (len(unique_x) - 1) if len(unique_x) > 1 else 1.0
    step_y = (max_y - min_y) / (len(unique_y) - 1) if len(unique_y) > 1 else 1.0

    col_indices = np.round((centers[:, 0] - min_x) / step_x).astype(int)
    row_indices = np.round((centers[:, 1] - min_y) / step_y).astype(int)

    n_rows = row_indices.max() + 1
    n_cols = col_indices.max() + 1

    cell_correlations = np.full((n_rows, n_cols), np.nan)

    for k, cell in enumerate(cells):
        r, c = row_indices[k], col_indices[k]
        cell_correlations[r, c] = cell.best_score

    return cell_correlations
