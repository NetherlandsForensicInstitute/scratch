"""
Alignment functions for striation marks.

This module provides functions for detecting striation direction and
aligning striated marks to be horizontal.
"""

import numpy as np
from numpy.typing import NDArray

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType
from conversion.filter import (
    gaussian_sigma_to_cutoff,
    apply_gaussian_regression_filter,
)
from conversion.mask import crop_to_mask, _determine_bounding_box
from conversion.resample import resample_scan_image_and_mask
from conversion.preprocess_striation.shear import shear_data_by_shifting_profiles

# Maximum expected striation angle for outlier filtering in gradient detection (degrees)
_MAX_GRADIENT_ANGLE_DEG = 10.0


def fine_align_bullet_marks(
    scan_image: ScanImage,
    mark_type: MarkType | None = None,
    mask: MaskArray | None = None,
    angle_accuracy: float = 0.1,
    cut_y_after_shift: bool = True,
    max_iter: int = 25,
    subsampling_factor: int = 1,
) -> tuple[ScanImage, MaskArray | None, float]:
    """
    Fine alignment of striated marks by iteratively detecting striation direction.

    Iteratively determines the direction of striation marks, and shear transforms the
    depth data so that striations are horizontal.

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param mark_type: Mark type enum value (optional, for resampling).
    :param mask: Optional boolean mask (True = valid).
    :param angle_accuracy: Target angle accuracy in degrees (default 0.1).
    :param cut_y_after_shift: If True, crop borders after shifting.
    :param max_iter: Maximum number of iterations.
    :param subsampling_factor: Additional subsampling factor for gradient detection.
    :returns: Tuple of (aligned_scan_image, aligned_mask, total_angle_degrees).
    """
    # Find alignment angle iteratively
    total_angle = _find_alignment_angle(
        scan_image,
        mask,
        angle_accuracy,
        cut_y_after_shift,
        max_iter,
        subsampling_factor,
    )

    # Apply shear transformation if angle is non-zero
    if not np.isclose(total_angle, 0.0, atol=1e-09):
        total_angle_rad = np.radians(total_angle)
        result_data = shear_data_by_shifting_profiles(
            scan_image.data, total_angle_rad, cut_y_after_shift
        )
        if mask is not None:
            result_mask = (
                shear_data_by_shifting_profiles(
                    mask, total_angle_rad, cut_y_after_shift
                )
                > 0.5
            )
        else:
            result_mask = None
    else:
        result_data = scan_image.data
        result_mask = mask

    # Crop to mask bounding box
    if result_mask is not None:
        result_data = crop_to_mask(result_data, result_mask)
        y_slice, x_slice = _determine_bounding_box(result_mask)
        result_mask = result_mask[y_slice, x_slice]

    result_scan = ScanImage(
        data=np.asarray(result_data, dtype=np.float64),
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )

    # Resample to mark type target scale if specified
    if mark_type is not None:
        # TODO: possible the resampling is not needed as the images are already resamples before preprocessing_striation
        result_scan, result_mask = resample_scan_image_and_mask(
            result_scan,
            result_mask,
            target_scale=mark_type.scale,
            only_downsample=True,
        )

    return result_scan, result_mask, total_angle


def _find_alignment_angle(
    scan_image: ScanImage,
    mask: MaskArray | None,
    angle_accuracy: float,
    cut_y_after_shift: bool,
    max_iter: int,
    subsampling_factor: int,
) -> float:
    """
    Iteratively find the alignment angle for striation marks.

    The alignment angle is the rotation needed to make striations horizontal.
    Positive angle means clockwise rotation.

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param mask: Optional boolean mask.
    :param angle_accuracy: Target angle accuracy in degrees.
    :param cut_y_after_shift: If True, crop borders after shifting.
    :param max_iter: Maximum number of iterations.
    :param subsampling_factor: Additional subsampling factor.
    :returns: Total alignment angle in degrees (0.0 if max_iter reached).
    """
    data_tmp = scan_image.data
    mask_tmp = mask

    total_angle = 0.0
    current_angle = -45.0  # Initialize with large angle to ensure first iteration runs
    previous_angle = 0.0

    # Iteratively refine the alignment angle until striations are horizontal enough.
    # Each iteration detects the remaining misalignment, accumulates it into total_angle,
    # then re-rotates the original data by total_angle and re-measures.
    for _ in range(max_iter):
        # Check convergence
        if abs(current_angle) <= angle_accuracy:
            break

        tmp_scan_image = scan_image.model_copy(update={"data": data_tmp})
        current_angle = _detect_striation_angle(
            tmp_scan_image, mask=mask_tmp, subsampling_factor=subsampling_factor
        )

        if np.isnan(current_angle):
            current_angle = 0.05
        else:
            total_angle += current_angle
            total_angle_rad = np.radians(total_angle)
            data_tmp = shear_data_by_shifting_profiles(
                scan_image.data, total_angle_rad, cut_y_after_shift
            )
            if mask is not None:
                mask_tmp = (
                    shear_data_by_shifting_profiles(
                        mask, total_angle_rad, cut_y_after_shift
                    )
                    > 0.5
                )
            else:
                mask_tmp = None

            # Check if stuck (same angle as previous iteration)
            if current_angle == previous_angle:
                break
            previous_angle = current_angle
    else:
        # Max iterations reached without convergence
        return 0.0

    return total_angle


def _detect_striation_angle(
    scan_image: ScanImage,
    mask: MaskArray | None = None,
    subsampling_factor: int = 1,
) -> float:
    """
    Determine the striation direction using gradient analysis.

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param mask: Optional boolean mask (True = valid).
    :param subsampling_factor: Additional subsampling factor on top of the automatic
        subsampling for faster calculation, but lower precision.
    :returns: Detected striation angle in degrees (positive = clockwise),
        in the range (-10, 10), or NaN if no valid gradients are found.
    """
    # Determine subsampling factor
    sub_samp = subsampling_factor
    if scan_image.scale_x < 1e-6:
        sub_samp = round(1e-6 / scan_image.scale_x) * subsampling_factor

    # Resample data (only x-dimension, matching MATLAB's resample function)
    if sub_samp > 1 and scan_image.width // sub_samp >= 2:
        scan_image, mask = resample_scan_image_and_mask(
            scan_image,
            mask,
            factors=(sub_samp, 1),
            only_downsample=True,
        )

    # Smooth data using Gaussian filter (MATLAB-equivalent)
    # Original: sigma = max(3, round(1.75e-5 / effective_pixel_size)) in pixels
    effective_pixel_size = scan_image.scale_x
    sigma_pixels = max(3, round(1.75e-5 / effective_pixel_size))
    smoothing_cutoff = gaussian_sigma_to_cutoff(sigma_pixels, effective_pixel_size)

    smoothed = apply_gaussian_regression_filter(
        scan_image.data,
        cutoff_length=smoothing_cutoff,
        pixel_size=(effective_pixel_size, effective_pixel_size),
        regression_order=0,
        nan_out=True,
    )

    # Calculate gradient and extract tilt angles
    fy, fx = np.gradient(smoothed)
    angles = _compute_tilt_angles_from_gradient(fx, fy, mask)

    # Filter outliers: keep only angles within expected range for fine alignment
    angles = angles[np.abs(angles) < np.radians(_MAX_GRADIENT_ANGLE_DEG)]

    if len(angles) == 0:
        return np.nan

    return float(np.degrees(np.median(angles)))


def _compute_tilt_angles_from_gradient(
    fx: NDArray[np.floating],
    fy: NDArray[np.floating],
    mask: MaskArray | None = None,
) -> NDArray[np.floating]:
    """
    Compute striation tilt angles from gradient components.

    For horizontal striations, gradients point vertically (fx=0, fy≠0).
    Tilted striations produce a horizontal gradient component: sin(θ) = fx/|grad|.

    :param fx: Gradient in x-direction.
    :param fy: Gradient in y-direction.
    :param mask: Optional boolean mask (True = valid pixels to include).
    :returns: Array of tilt angles in radians for valid pixels.
    """
    # Calculate total gradient (L1 norm)
    grad_magnitude = np.abs(fx) + np.abs(fy)

    # Create gradient threshold mask (keep only significant gradients)
    grad_threshold = 1.5 * np.nanmedian(grad_magnitude)
    grad_mask = grad_magnitude > grad_threshold

    # Combine with input mask if provided
    if mask is not None:
        grad_mask = grad_mask & (mask > 0.5)

    # Normalize fx by gradient magnitude
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_norm = fx / grad_magnitude

    # Ensure consistent sign by aligning with fy direction
    fx_norm = fx_norm * np.sign(fy)

    # Extract angles for valid pixels
    fx_flat = fx_norm.flatten()
    mask_flat = grad_mask.flatten()

    return np.arcsin(np.clip(fx_flat[mask_flat], -1, 1))
