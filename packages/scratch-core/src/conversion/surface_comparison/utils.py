from collections.abc import Sequence

import numpy as np
from loguru import logger

from container_models.base import FloatArray1D, FloatArray2D, Points2D
from container_models.scan_image import ScanImage
from conversion.exceptions import ImageNotIsotropicError
from conversion.surface_comparison.models import Cell, ComparisonParams

# Tolerances for np.isclose() when comparing pixel scales (e.g. isotropy check, matching scales between images).
SCALE_COMPARISON_ATOL = 0.0
SCALE_COMPARISON_RTOL = 1e-3


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


def wrap_angles(angles: FloatArray1D) -> FloatArray1D:
    """
    Normalize angles in radians to the [-pi, pi] interval.

    :param angles: Array of angles in radians.
    :returns: Array of normalized angles in radians.
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def _cells_correlation_to_grid(cells: Sequence[Cell]) -> FloatArray2D:
    """
    Map unordered cells onto a row-major grid with the correlation as values.

    Grid indices are derived from the cell centers and the cell pitch. Pitch is taken from the
    smallest spacing between distinct coordinates to avoid collapsing rows when the cell layout
    contains gaps (e.g. a breech-face annulus).

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


def assert_image_is_isotropic(scan_image: ScanImage) -> None:
    if not np.isclose(
        scan_image.scale_x,
        scan_image.scale_y,
        atol=SCALE_COMPARISON_ATOL,
        rtol=SCALE_COMPARISON_RTOL,
    ):
        raise ImageNotIsotropicError(
            scale_x=scan_image.scale_x, scale_y=scan_image.scale_y
        )


def resolve_nan_fill_value(
    reference_image: ScanImage, params: ComparisonParams
) -> float | None:
    """
    Turn ``template_nan_fill_strategy`` into the concrete value every template will be filled with.

    ``None`` means "each cell's own valid-pixel mean"; see conversion.surface_comparison.template_fill.fill_template_nan.

    :param reference_image: Reference scan image; its global mean is used for ``global_mean`` strategy.
    :param params: Comparison parameters specifying the fill strategy.
    :returns: A fill value for ``global_mean``, or ``None`` for ``local_mean``.
    """
    # TODO: ``local_mean`` needs the masked NCC of Padfield (2012) to be correct. The score denominator in
    #  conversion.surface_comparison.cell_registration.scoring.build_correlation_basis normalizes over the whole window,
    #  while the numerator only covers the overlap of the two validity masks, so scores are deflated in proportion to
    #  how empty a cell is. ``global_mean`` wins today because it happens to offset that.
    if params.template_nan_fill_strategy != "global_mean":
        return None
    nan_fill_value = float(np.nanmean(reference_image.data))
    logger.debug(
        "Using global mean ({:.4f}) for NaN filling (template_nan_fill_strategy=global_mean)",
        nan_fill_value,
    )
    return nan_fill_value
