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

    Grid indices are derived from the cell centers and the cell pitch. The
    pitch is taken from the smallest spacing between distinct center
    coordinates rather than from (max - min) / (n_unique - 1): surfaces with
    holes (e.g. a breech-face annulus) have gaps in the cell layout, and
    averaging across a gap yields a too-large step that collapses distinct
    cell rows onto one grid row.

    :param cells: Unordered cell results from the CMC pipeline.
    :return: cell_correlations (n_rows, n_cols), NaN where there is no cell.
    """
    centers = np.array([cell.center_reference for cell in cells])

    unique_x = np.unique(np.round(centers[:, 0], decimals=9))
    unique_y = np.unique(np.round(centers[:, 1], decimals=9))
    min_x, min_y = unique_x[0], unique_y[0]

    step_x = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 1.0
    step_y = np.min(np.diff(unique_y)) if len(unique_y) > 1 else 1.0

    col_indices = np.round((centers[:, 0] - min_x) / step_x).astype(int)
    row_indices = np.round((centers[:, 1] - min_y) / step_y).astype(int)

    n_rows = row_indices.max() + 1
    n_cols = col_indices.max() + 1

    cell_correlations = np.full((n_rows, n_cols), np.nan)
    for k, cell in enumerate(cells):
        r, c = row_indices[k], col_indices[k]
        cell_correlations[r, c] = cell.best_score

    n_scored = sum(not np.isnan(cell.best_score) for cell in cells)
    if np.count_nonzero(~np.isnan(cell_correlations)) != n_scored:
        raise ValueError("cell centers do not map onto a unique grid position")

    return cell_correlations
