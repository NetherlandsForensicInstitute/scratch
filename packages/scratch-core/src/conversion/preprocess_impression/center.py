import numpy as np
from numpy._typing import NDArray
from scipy.ndimage import binary_erosion
from skimage.measure import ransac, CircleModel

from container_models.base import MaskArray
from conversion.data_formats import Mark, MarkType
from conversion.mask import get_bounding_box
from conversion.preprocess_impression.utils import Point2D


def _get_mask_inner_edge_points(mask: MaskArray) -> NDArray[np.floating]:
    """
    Extract inner edge points of a binary mask.

    :param mask: Binary mask array.
    :return: Array of (x, y) edge points in pixel coordinates.
    """
    eroded = binary_erosion(mask).astype(bool)
    edge = mask & ~eroded
    rows, cols = np.where(edge)
    return np.column_stack([cols, rows]).astype(float)


def _points_are_collinear(points: NDArray[np.floating], tol: float = 1e-9) -> bool:
    """
    Check if points are approximately collinear using SVD.

    :param points: Array of 2D points. Shape: (n_points, 2)
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


def _get_bounding_box_center(mask: MaskArray) -> Point2D:
    """
    Compute center of bounding box for True values in mask.

    :param mask: Boolean mask array.
    :return: Center (x, y) in pixel coordinates.
    """
    x_slice, y_slice = get_bounding_box(mask)
    return (
        (x_slice.start + x_slice.stop) / 2,
        (y_slice.start + y_slice.stop) / 2,
    )


def _compute_map_center(
    array: MaskArray,
    use_circle_fit: bool = False,
) -> Point2D:
    """
    Compute map center from data bounds or circle fit.

    :param array: boolean array to get the center for.
    :param use_circle_fit: If True, attempt RANSAC circle fitting first.
    :return: Center position (x, y) in pixel coordinates.
    """
    if use_circle_fit:
        edge_points = _get_mask_inner_edge_points(array)
        if (center := _fit_circle_ransac(edge_points)) is not None:
            return center

    return _get_bounding_box_center(array)


def compute_center_local(mark: Mark) -> Point2D:
    """
    Compute center of mark data in local (image) coordinates.

    For breech face impressions, uses RANSAC circle fitting on the data
    boundary (these marks are typically circular). For other mark types, uses the center of the bounding box
    of valid (non-NaN) data.

    :param mark: Input mark containing scan image and mark type.
    :return: Center (x, y) in meters, relative to image origin (top-left).
    """
    use_circle = mark.mark_type == MarkType.BREECH_FACE_IMPRESSION
    cx, cy = _compute_map_center(mark.scan_image.valid_mask, use_circle_fit=use_circle)
    return (
        cx * mark.scan_image.scale_x,
        cy * mark.scan_image.scale_y,
    )
