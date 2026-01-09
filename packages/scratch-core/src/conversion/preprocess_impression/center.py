import numpy as np
from numpy._typing import NDArray
from scipy.ndimage import binary_erosion
from skimage.measure import ransac, CircleModel

from container_models.base import MaskArray, ScanMap2DArray
from conversion.data_formats import Mark, MarkType
from conversion.preprocess_impression.utils import Point2D


def _get_mask_edge_points(mask: MaskArray) -> NDArray[np.floating]:
    """
    Extract inner edge points of a binary mask.

    :param mask: Binary mask array.
    :return: Array of (col, row) edge points in pixel coordinates.
    """
    eroded = binary_erosion(mask).astype(bool)
    edge = mask & ~eroded
    rows, cols = np.where(edge)
    return np.column_stack([cols, rows]).astype(float)


def _points_are_collinear(points: NDArray[np.floating], tol: float = 1e-9) -> bool:
    """
    Check if points are approximately collinear using SVD.

    :param points: Array of 2D points.
    :param tol: Tolerance for collinearity check.
    :return: True if points are collinear.
    """
    if len(points) < 3:
        return True
    centered = points - points.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    return singular_values[-1] < tol * singular_values[0]


def _fit_circle_ransac(
    points: NDArray[np.floating],
    n_iterations: int = 1000,
    threshold: float = 1.0,
) -> Point2D | None:
    """
    Fit a circle to 2D points using RANSAC.

    :param points: Array of (x, y) points with shape (N, 2).
    :param n_iterations: Maximum RANSAC iterations.
    :param threshold: Inlier distance threshold in same units as points.
    :return: Circle center (x, y) or None if fitting failed.
    :raises ValueError: If points array has wrong shape.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array, got {points.shape}")

    if _points_are_collinear(points):
        return None

    model, _ = ransac(
        points,
        CircleModel,
        min_samples=3,
        residual_threshold=threshold,
        max_trials=n_iterations,
    )

    if model is None:
        return None

    x, y, radius = model.params
    if radius > 0 and np.isfinite([x, y, radius]).all():
        return x, y

    return None


def _get_bounding_box_center(mask: NDArray[np.bool_]) -> Point2D:
    """
    Compute center of bounding box for True values in mask.

    :param mask: Boolean mask array.
    :return: Center (x, y) in pixel coordinates.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return mask.shape[1] / 2, mask.shape[0] / 2
    return (
        (cols.min() + cols.max() + 1) / 2,
        (rows.min() + rows.max() + 1) / 2,
    )


def _compute_map_center(
    data: ScanMap2DArray,
    use_circle_fit: bool = False,
) -> Point2D:
    """
    Compute map center from data bounds or circle fit.

    :param data: Height map array.
    :param use_circle_fit: If True, attempt RANSAC circle fitting first.
    :return: Center position (col, row) in pixel coordinates.
    """
    valid_mask = ~np.isnan(data)

    if use_circle_fit:
        edge_points = _get_mask_edge_points(valid_mask)
        if (center := _fit_circle_ransac(edge_points)) is not None:
            return center

    return _get_bounding_box_center(valid_mask)


def compute_center_local(mark: Mark) -> Point2D:
    """
    Compute local center coordinates from mark data.

    :param mark: Input mark image.
    :return: Center (x, y) in meters.
    """
    use_circle = mark.mark_type == MarkType.BREECH_FACE_IMPRESSION
    center_px = _compute_map_center(mark.scan_image.data, use_circle_fit=use_circle)
    return (
        center_px[0] * mark.scan_image.scale_x,
        center_px[1] * mark.scan_image.scale_y,
    )
