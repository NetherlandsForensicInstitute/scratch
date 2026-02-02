"""
Shear transformation functions for striation alignment.

This module provides geometric transformations for aligning striated marks
by shifting vertical profiles.
"""

import numpy as np
from skimage.transform import AffineTransform, warp

from container_models.base import DepthData, FloatArray2D


def shear_data_by_shifting_profiles(
    depth_data: DepthData,
    angle_rad: float,
    cut_y_after_shift: bool = True,
) -> FloatArray2D:
    """
    Shear depth data by shifting each column (profile) vertically.

    This implements a shear transformation that preserves striation features.
    Instead of true 2D rotation (which would require interpolation in both
    dimensions), each column is shifted up or down by an amount proportional
    to its x-position. This creates a rotation effect while keeping each
    vertical profile intact.

    Example for a 5° (0.087 rad) shear on a 100-pixel wide image:
        - Left edge (col 0): shifts up by ~4.4 pixels
        - Center (col 50): no shift
        - Right edge (col 99): shifts down by ~4.4 pixels

    :param depth_data: 2D depth data array (rows x cols).
    :param angle_rad: Shear angle in radians (positive = clockwise).
        Expected range: ±0.175 rad (±10°). Angles < 0.00175 rad (0.1°) are skipped.
    :param cut_y_after_shift: If True, crop NaN borders introduced by shifting.
    :returns: The sheared depth data.
    """
    # Skip shear for angles smaller than ~0.1° (0.00175 rad)
    if abs(angle_rad) <= 0.00175:
        return depth_data.astype(np.floating).copy()

    height, width = depth_data.shape
    center_x, center_y = width / 2, height / 2

    # Calculate padding needed to accommodate the shear
    total_shift = abs(np.tan(angle_rad) * width)
    padding = int(np.ceil(total_shift / 2)) + 2

    # Pad data with NaN to accommodate shear
    padded_height = height + 2 * padding
    padded_data = np.full((padded_height, width), np.nan)
    padded_data[padding : padding + height, :] = depth_data
    padded_center_y = center_y + padding

    # Build shear transform centered at the middle of the image:
    # 1. Translate center to origin
    # 2. Apply vertical shear (shift y based on x position)
    # 3. Translate back
    translate_to_origin = AffineTransform(translation=(-center_x, -padded_center_y))
    shear_transform = AffineTransform(shear=(0, angle_rad))  # Vertical shear
    translate_back = AffineTransform(translation=(center_x, padded_center_y))
    combined = translate_to_origin + shear_transform + translate_back

    # Apply the transformation
    output = warp(
        padded_data,
        inverse_map=combined.inverse,
        order=1,  # Bilinear interpolation
        mode="constant",
        cval=np.nan,
        preserve_range=True,
    )

    # Crop NaN borders if requested
    if cut_y_after_shift:
        num_nan_rows = int(np.ceil(total_shift / 2)) + 1
        crop_start = padding + num_nan_rows
        crop_end = padded_height - padding - num_nan_rows
        output = output[crop_start:crop_end, :]

    return output


def propagate_nan(data: FloatArray2D) -> FloatArray2D:
    """
    Propagate NaN values to adjacent pixels in the down and right directions.

    This matches MATLAB's asymmetric NaN propagation behavior where pixels
    immediately above or to the left of NaN regions are also set to NaN.
    The asymmetry comes from MATLAB's filtering direction (y-axis first,
    then x-axis) using causal boundary handling.

    :param data: 2D array with potential NaN values.
    :returns: Array with NaN propagated to up/left neighbors of NaN regions.
    """
    if not np.any(np.isnan(data)):
        return data

    result = data.copy()
    nan_mask = np.isnan(data)

    # Dilate NaN mask: if position (r+1, c) is NaN, then (r, c) becomes NaN
    # This means NaN propagates upward from NaN regions
    dilated = nan_mask.copy()

    # Down: if NaN at (r+1, c), set NaN at (r, c)
    dilated[:-1, :] |= nan_mask[1:, :]
    # Right: if NaN at (r, c+1), set NaN at (r, c)
    dilated[:, :-1] |= nan_mask[:, 1:]

    # Apply dilated mask
    result[dilated] = np.nan
    return result
