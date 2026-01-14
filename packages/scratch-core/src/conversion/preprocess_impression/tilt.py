from typing import NamedTuple

import numpy as np
from numpy._typing import NDArray

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.preprocess_impression.utils import update_mark_scan_image, Point2D


class TiltEstimate(NamedTuple):
    """Result of plane tilt estimation."""

    tilt_x_rad: float
    tilt_y_rad: float
    residuals: NDArray[np.floating]


def apply_tilt_correction(
    mark: Mark,
    center_local: Point2D,
) -> tuple[Mark, Point2D]:
    """
    Apply tilt correction if requested.

    :param mark: Input mark.
    :param center_local: Local center coordinates.
    :return: Tuple of (corrected mark, updated center).
    """
    adjusted_scan, center_local = _adjust_for_plane_tilt(mark.scan_image, center_local)
    return update_mark_scan_image(mark, adjusted_scan), center_local


def _estimate_plane_tilt(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
) -> TiltEstimate:
    """
    Estimate best-fit plane tilt angles using least squares.

    Fits z = ax + by + c to the data points. The tilt angles represent the
    inclination of the fitted plane relative to the horizontal (x-y) plane:
    tilt_x is the angle along the x-axis, tilt_y along the y-axis.

    :param x: X coordinates.
    :param y: Y coordinates.
    :param z: Z values at each (x, y).
    :return: TiltEstimate with tilt angles in degrees and residuals.
    """
    design_matrix = np.column_stack([x, y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(design_matrix, z, rcond=None)

    return TiltEstimate(
        tilt_x_rad=np.arctan(a),
        tilt_y_rad=np.arctan(b),
        residuals=z - (a * x + b * y + c),
    )


def _get_valid_coordinates(
    scan_image: ScanImage,
    center: Point2D,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Extract x, y, z coordinates of valid pixels, centered at origin.

    :param scan_image: Input scan image.
    :param center: Center point to subtract (in meters).
    :return: Tuple of (x, y, z) coordinate arrays in meters.
    """
    rows, cols = np.where(scan_image.valid_mask)

    xs = cols * scan_image.scale_x - center[0]
    ys = rows * scan_image.scale_y - center[1]
    zs = scan_image.valid_data

    return xs, ys, zs


def _adjust_for_plane_tilt(
    scan_image: ScanImage,
    center: Point2D,
) -> tuple[ScanImage, Point2D]:
    """
    Remove plane tilt from scan image and adjust scale factors.

    :param scan_image: Input scan image.
    :param center: Center position in meters.
    :return: Tuple of (leveled scan image, adjusted center).
    :raises ValueError: If fewer than 3 valid points exist.
    """
    xs, ys, zs = _get_valid_coordinates(scan_image, center)

    if len(xs) < 3:
        raise ValueError("Need at least 3 valid points to estimate plane tilt")

    tilt = _estimate_plane_tilt(xs, ys, zs)

    # Update data with residuals
    data = scan_image.data.copy()
    data[scan_image.valid_mask] = tilt.residuals

    # Adjust scales for tilt
    cos_x = np.cos(tilt.tilt_x_rad)
    cos_y = np.cos(tilt.tilt_y_rad)
    scale_x_new = scan_image.scale_x / cos_x
    scale_y_new = scan_image.scale_y / cos_y

    # Adjust center for new scales
    center_new = (
        center[0] * scale_x_new / scan_image.scale_x,
        center[1] * scale_y_new / scan_image.scale_y,
    )

    return ScanImage(data=data, scale_x=scale_x_new, scale_y=scale_y_new), center_new
