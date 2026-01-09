from typing import NamedTuple

import numpy as np
from numpy._typing import NDArray

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.preprocess_impression.utils import _update_mark_scan_image, Point2D


class TiltEstimate(NamedTuple):
    """Result of plane tilt estimation."""

    tilt_x_deg: float
    tilt_y_deg: float
    residuals: NDArray[np.floating]


def _estimate_plane_tilt(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
) -> TiltEstimate:
    """
    Estimate best-fit plane tilt angles using least squares.

    Fits z = ax + by + c to the data points.

    :param x: X coordinates.
    :param y: Y coordinates.
    :param z: Z values at each (x, y).
    :return: TiltEstimate with tilt angles in degrees and residuals.
    """
    design_matrix = np.column_stack([x, y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(design_matrix, z, rcond=None)

    return TiltEstimate(
        tilt_x_deg=np.degrees(np.arctan(a)),
        tilt_y_deg=np.degrees(np.arctan(b)),
        residuals=z - (a * x + b * y + c),
    )


def _get_valid_coordinates(
    scan_image: ScanImage,
    center: Point2D,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Extract x, y, z coordinates of valid pixels, centered at origin.

    :param scan_image: Input scan image.
    :param center: Center point to subtract (in pixel coordinates).
    :return: Tuple of (x, y, z) coordinate arrays in meters.
    """
    valid_mask = ~np.isnan(scan_image.data)
    rows, cols = np.where(valid_mask)

    x = cols * scan_image.scale_x - center[0]
    y = rows * scan_image.scale_y - center[1]
    z = scan_image.data[valid_mask]

    return x, y, z


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
    x, y, z = _get_valid_coordinates(scan_image, center)

    if len(x) < 3:
        raise ValueError("Need at least 3 valid points to estimate plane tilt")

    tilt = _estimate_plane_tilt(x, y, z)

    # Update data with residuals
    data = scan_image.data.copy()
    data[~np.isnan(scan_image.data)] = tilt.residuals

    # Adjust scales for tilt
    cos_x = np.cos(np.radians(tilt.tilt_x_deg))
    cos_y = np.cos(np.radians(tilt.tilt_y_deg))
    scale_x_new = scan_image.scale_x / cos_x
    scale_y_new = scan_image.scale_y / cos_y

    # Adjust center for new scales
    center_new = (
        center[0] * scale_x_new / scan_image.scale_x,
        center[1] * scale_y_new / scan_image.scale_y,
    )

    return ScanImage(data=data, scale_x=scale_x_new, scale_y=scale_y_new), center_new


def apply_tilt_correction(
    mark: Mark,
    center_local: Point2D,
    should_adjust: bool,
) -> tuple[Mark, Point2D]:
    """
    Apply tilt correction if requested.

    :param mark: Input mark.
    :param center_local: Local center coordinates.
    :param should_adjust: Whether to apply tilt correction.
    :return: Tuple of (corrected mark, updated center).
    """
    if not should_adjust:
        return mark, center_local

    adjusted_scan, center_local = _adjust_for_plane_tilt(mark.scan_image, center_local)
    return _update_mark_scan_image(mark, adjusted_scan), center_local
