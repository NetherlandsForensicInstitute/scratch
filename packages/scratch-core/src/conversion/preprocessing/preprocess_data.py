"""Preprocessing pipeline for striated tool and bullet marks.

This module implements the PreprocessData pipeline from MATLAB:
- Step 2: Form and noise removal (shape removal via highpass, noise removal via lowpass)
- Step 3: Fine rotation to align striations horizontally + profile extraction

Translated from MATLAB: PreprocessData.m, FineAlignBulletMarks.m, RotateImageGradVector.m,
RotateDataByShiftingProfiles.m, RemoveZeroImageBorder.m, ResampleMarkTypeSpecific.m
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import interp1d

from conversion.preprocessing.preprocess_data_filter import (
    apply_gaussian_filter_1d,
    cheby_cutoff_to_gauss_sigma,
)


# =============================================================================
# Step 2: Form and Noise Removal
# =============================================================================


def apply_shape_noise_removal(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff_hi: float,
    mask: NDArray[np.bool_] | None = None,
    cut_borders_after_smoothing: bool = True,
    cutoff_lo: float = 250e-6,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Apply large-scale shape and noise removal to isolate striation features.

    The function has the following steps:

    **Step 1: Calculate sigma and check data size**
        Convert the cutoff wavelength to Gaussian sigma. If the data is
        too short (2*sigma > 20% of height), disable border cutting to
        preserve data.

    **Step 2: Shape removal**
        Use apply_gaussian_filter_1d with is_high_pass=True to remove
        large-scale shape (curvature, tilt, waviness).

    **Step 3: Noise removal**
        Apply apply_gaussian_filter_1d with is_high_pass=False (lowpass)
        to remove high-frequency noise while preserving striation features.


    :param depth_data: 2D depth/height data array. Should already be coarsely aligned.
    :param xdim: Pixel spacing in meters (m).
    :param cutoff_hi: High-frequency cutoff wavelength in meters (m) for shape removal.
    :param mask: Boolean mask array (True = valid data).
    :param cut_borders_after_smoothing: If True, crop filter edge artifacts.
    :param cutoff_lo: Low-frequency cutoff wavelength in meters (m) for noise removal.

    :returns: Tuple of (processed_data, mask).
    """
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(depth_data.shape, dtype=bool)

    # Calculate Gaussian sigma from cutoff wavelength
    sigma = cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)

    # Check if data is too short for border cutting
    data_height = depth_data.shape[0]
    data_too_short = (2 * sigma) > (data_height * 0.2)

    # Override border cutting for short data
    if data_too_short:
        cut_borders = False
    else:
        cut_borders = cut_borders_after_smoothing

    # Shape removal (highpass filter)
    data_no_shape, _, mask_shape = apply_gaussian_filter_1d(
        depth_data,
        xdim=xdim,
        cutoff=cutoff_hi,
        is_high_pass=True,
        cut_borders_after_smoothing=cut_borders,
        mask=mask,
    )

    # Noise removal (lowpass filter)
    data_no_noise, _, mask_noise = apply_gaussian_filter_1d(
        data_no_shape,
        xdim=xdim,
        cutoff=cutoff_lo,
        is_high_pass=False,
        cut_borders_after_smoothing=cut_borders,
        mask=mask_shape,
    )

    return data_no_noise, mask_noise


# =============================================================================
# Step 3: Fine Alignment Helper Functions
# =============================================================================


def _smooth_2d(
    data: NDArray[np.floating],
    sigma: float | tuple[float, float],
) -> NDArray[np.floating]:
    """
    Apply 2D Gaussian smoothing to data with NaN handling.

    :param data: 2D array to smooth.
    :param sigma: Gaussian sigma (scalar or (sigma_row, sigma_col)).
    :returns: Smoothed data array.
    """
    if np.any(np.isnan(data)):
        # NaN-aware smoothing: replace NaN with 0, track weights
        mask = ~np.isnan(data)
        data_filled = np.where(mask, data, 0.0)
        weights = mask.astype(float)

        smoothed_data = gaussian_filter(data_filled, sigma, mode="nearest")
        smoothed_weights = gaussian_filter(weights, sigma, mode="nearest")

        with np.errstate(divide="ignore", invalid="ignore"):
            result = smoothed_data / smoothed_weights
            result[smoothed_weights == 0] = np.nan

        return result
    else:
        return gaussian_filter(data, sigma, mode="nearest")


def _remove_zero_image_border(
    data: NDArray[np.floating],
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Remove zero/invalid borders from masked data.

    :param data: 2D depth data array.
    :param mask: Boolean mask (True = valid).
    :returns: Tuple of (cropped_data, cropped_mask).
    """
    # Find row bounds
    row_sum = np.sum(mask, axis=1)
    valid_rows = np.nonzero(row_sum > 0)[0]
    if len(valid_rows) == 0:
        return data, mask
    start_row, end_row = valid_rows[0], valid_rows[-1] + 1

    # Find column bounds
    col_sum = np.sum(mask, axis=0)
    valid_cols = np.nonzero(col_sum > 0)[0]
    if len(valid_cols) == 0:
        return data, mask
    start_col, end_col = valid_cols[0], valid_cols[-1] + 1

    # Apply mask to data and crop
    data_masked = data * mask.astype(float)
    cropped_data = data_masked[start_row:end_row, start_col:end_col]
    cropped_mask = mask[start_row:end_row, start_col:end_col]

    return cropped_data, cropped_mask


def _rotate_data_by_shifting_profiles(
    depth_data: NDArray[np.floating],
    angle: float,
    cut_y_after_shift: bool = True,
) -> NDArray[np.floating]:
    """
    Rotate depth data by shifting profiles (columns) vertically.

    :param depth_data: 2D depth data array (rows x cols).
    :param angle: Rotation angle in degrees.
    :param cut_y_after_shift: If True, crop borders after shifting.
    :returns: Rotated depth data.
    """
    if abs(angle) <= 0.1:
        return depth_data.copy()

    depth_data = depth_data.astype(float)
    height, width = depth_data.shape

    # Calculate total vertical shift across the image
    total_shift = np.tan(np.radians(angle)) * width

    # Calculate shift for each column
    shift_y = np.linspace(total_shift / 2, -total_shift / 2, width)

    # Amount of padding needed
    amount_zeros = int(np.ceil(abs(total_shift) / 2))
    padding = amount_zeros + 2

    # Pad data with NaN
    padded_height = height + 2 * padding
    depth_data_new = np.full((padded_height, width), np.nan)
    depth_data_new[padding : padding + height, :] = depth_data

    # Shift each column
    original_indices = np.arange(padded_height)
    for col in range(width):
        col_data = depth_data_new[:, col]
        valid_mask = ~np.isnan(col_data)

        if np.sum(valid_mask) > 1:
            interp_func = interp1d(
                original_indices[valid_mask],
                col_data[valid_mask],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            new_indices = original_indices - shift_y[col]
            depth_data_new[:, col] = interp_func(new_indices)

    # Crop borders if requested
    if cut_y_after_shift:
        amount_of_nans = int(np.ceil(max(abs(shift_y[0]), abs(shift_y[-1])))) + 1
        crop_start = padding + amount_of_nans
        crop_end = padded_height - padding - amount_of_nans
        depth_data_new = depth_data_new[crop_start:crop_end, :]

    return depth_data_new


def _rotate_image_grad_vector(
    depth_data: NDArray[np.floating],
    xdim: float,
    mask: NDArray[np.bool_] | None = None,
    extra_sub_samp: int = 1,
) -> float:
    """
    Determine striation direction using gradient analysis.

    :param depth_data: 2D depth data array.
    :param xdim: Pixel spacing in meters.
    :param mask: Optional boolean mask (True = valid).
    :param extra_sub_samp: Additional subsampling factor.
    :returns: Detected rotation angle in degrees.
    """
    # Determine subsampling factor
    if xdim < 1e-6:
        sub_samp = int(round(1e-6 / xdim)) * extra_sub_samp
    else:
        sub_samp = 1 * extra_sub_samp

    # Determine sigma for smoothing
    sigma = round(10 * 1.75e-6 / xdim / sub_samp)
    if sigma < 3:
        sigma = 3

    # Subsample depth data
    if sub_samp > 1 and depth_data.shape[1] // sub_samp >= 2:
        depth_data_res = depth_data[:, ::sub_samp]
        if mask is not None:
            mask_res = mask[:, ::sub_samp]
        else:
            mask_res = None
    else:
        depth_data_res = depth_data
        mask_res = mask

    # Smooth data
    smoothed = _smooth_2d(depth_data_res, (sigma, sigma))

    # Calculate gradient
    fy, fx = np.gradient(smoothed)

    # Calculate total gradient magnitude
    grad_tmp = np.abs(fx) + np.abs(fy)

    # Create gradient threshold mask
    grad_threshold = 1.5 * np.nanmedian(grad_tmp)
    grad_mask = grad_tmp > grad_threshold

    # Combine with input mask if provided
    if mask_res is not None:
        grad_mask = grad_mask & (mask_res > 0.5)

    # Normalize fx by gradient magnitude
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_norm = fx / grad_tmp

    # Correct sign based on fy direction
    sign_correction = np.sign(fy)
    fx_norm = fx_norm * sign_correction

    # Flatten and apply mask
    fx_flat = fx_norm.flatten()
    mask_flat = grad_mask.flatten()

    # Calculate angles
    angles = np.degrees(np.arcsin(np.clip(fx_flat[mask_flat], -1, 1)))

    # Keep only angles between -10 and 10 degrees
    angles = angles[np.abs(angles) < 10]

    if len(angles) == 0:
        return np.nan

    return float(np.median(angles))


# Mark type to target sampling distance mapping (in meters)
MARK_TYPE_SAMPLING = {
    "Breech face impression": 3.5e-6,
    "Chamber impression": 1.5e-6,
    "Ejector impression": 1.5e-6,
    "Extractor impression": 1.5e-6,
    "Firing pin impression": 1.5e-6,
    "Aperture shear striation": 1.5e-6,
    "Bullet GEA striation": 1.5e-6,
    "Bullet LEA striation": 1.5e-6,
    "Chamber striation": 1.5e-6,
    "Ejector striation": 1.5e-6,
    "Ejector port striation": 1.5e-6,
    "Extractor striation": 1.5e-6,
    "Firing pin drag": 1.5e-6,
}


def _get_target_sampling_distance(mark_type: str) -> float:
    """
    Get target sampling distance for a given mark type.

    :param mark_type: Mark type string.
    :returns: Target sampling distance in meters.
    :raises ValueError: If mark type is not recognized.
    """
    for key, value in MARK_TYPE_SAMPLING.items():
        if key in mark_type:
            return value
    raise ValueError(f"Mark type not recognized: {mark_type}")


def _resample_mark_type_specific(
    depth_data: NDArray[np.floating],
    xdim: float,
    ydim: float,
    mark_type: str,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.floating], float, float, NDArray[np.bool_] | None]:
    """
    Resample depth data to target sampling distance based on mark type.

    :param depth_data: 2D depth data array.
    :param xdim: Current x-dimension pixel spacing in meters.
    :param ydim: Current y-dimension pixel spacing in meters.
    :param mark_type: Mark type string.
    :param mask: Optional boolean mask.
    :returns: Tuple of (resampled_data, new_xdim, new_ydim, resampled_mask).
    """
    target_sampling = _get_target_sampling_distance(mark_type)

    # Check if data meets minimum requirements
    if xdim > target_sampling and ydim > target_sampling:
        return depth_data, xdim, ydim, mask

    # Calculate resample factors
    resample_factor_x = xdim / target_sampling
    resample_factor_y = ydim / target_sampling

    # Resample depth data using bilinear interpolation
    depth_data_resampled = zoom(
        depth_data,
        (resample_factor_x, resample_factor_y),
        order=1,
        mode="nearest",
    )

    # Resample mask if provided (nearest neighbor)
    mask_resampled = None
    if mask is not None:
        mask_resampled = (
            zoom(
                mask.astype(float),
                (resample_factor_x, resample_factor_y),
                order=0,
                mode="nearest",
            )
            > 0.5
        )

    return depth_data_resampled, target_sampling, target_sampling, mask_resampled


# =============================================================================
# Step 3: Fine Alignment
# =============================================================================


def fine_align_bullet_marks(
    depth_data: NDArray[np.floating],
    xdim: float,
    ydim: float | None = None,
    mark_type: str | None = None,
    mask: NDArray[np.bool_] | None = None,
    angle_accuracy: float = 0.1,
    cut_y_after_shift: bool = True,
    max_iter: int = 25,
    extra_sub_samp: int = 1,
) -> tuple[NDArray[np.floating], NDArray[np.bool_] | None, float]:
    """
    Fine alignment of striated marks by iteratively detecting striation direction.

    Iteratively determines the direction of striation marks and rotates the
    depth data so that striations are horizontal.

    :param depth_data: 2D depth data array.
    :param xdim: X-dimension pixel spacing in meters.
    :param ydim: Y-dimension pixel spacing in meters (optional, for resampling).
    :param mark_type: Mark type string (optional, for resampling).
    :param mask: Optional boolean mask (True = valid).
    :param angle_accuracy: Target angle accuracy in degrees (default 0.1).
    :param cut_y_after_shift: If True, crop borders after shifting.
    :param max_iter: Maximum number of iterations.
    :param extra_sub_samp: Additional subsampling factor for gradient detection.
    :returns: Tuple of (aligned_data, aligned_mask, total_angle).
    """
    # Initialize
    data_tmp = depth_data.copy()
    mask_tmp = mask.copy() if mask is not None else None

    a_tot = 0.0
    a = -45.0
    a_last = 0.0
    iteration = 1

    # Iterative alignment
    while abs(a) > angle_accuracy and iteration < max_iter:
        a = _rotate_image_grad_vector(
            data_tmp,
            xdim,
            mask=mask_tmp,
            extra_sub_samp=extra_sub_samp,
        )

        if not np.isnan(a):
            a_tot = a_tot + a
            data_tmp = _rotate_data_by_shifting_profiles(
                depth_data, a_tot, cut_y_after_shift
            )

            if mask is not None:
                mask_float = mask.astype(float)
                mask_rotated = _rotate_data_by_shifting_profiles(
                    mask_float, a_tot, cut_y_after_shift
                )
                mask_tmp = mask_rotated > 0.5
                data_tmp, mask_tmp = _remove_zero_image_border(data_tmp, mask_tmp)

            if a == a_last:
                iteration = max_iter - 1
            else:
                a_last = a
        else:
            a = 0.05

        iteration += 1

    # Process result
    if iteration >= max_iter:
        a_tot = 0.0
        result_data = depth_data.copy()
        result_mask = mask
    else:
        result_data = _rotate_data_by_shifting_profiles(
            depth_data, a_tot, cut_y_after_shift
        )

        if mask is not None:
            mask_float = mask.astype(float)
            mask_rotated = _rotate_data_by_shifting_profiles(
                mask_float, a_tot, cut_y_after_shift
            )
            result_mask = mask_rotated > 0.5
            result_data, result_mask = _remove_zero_image_border(
                result_data, result_mask
            )

            if mark_type is not None and ydim is not None:
                result_data, _, _, result_mask = _resample_mark_type_specific(
                    result_data, xdim, ydim, mark_type, result_mask
                )
        else:
            result_mask = None
            if mark_type is not None and ydim is not None:
                result_data, _, _, _ = _resample_mark_type_specific(
                    result_data, xdim, ydim, mark_type
                )

    return result_data, result_mask, a_tot


def extract_profile(
    depth_data: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
    use_mean: bool = True,
) -> NDArray[np.floating]:
    """
    Extract a 1D profile from 2D depth data.

    Computes the mean or median profile along rows (axis 1).

    :param depth_data: 2D depth data array.
    :param mask: Optional boolean mask (True = valid).
    :param use_mean: If True, use mean; if False, use median.
    :returns: 1D profile array.
    """
    if mask is not None:
        masked_data = np.where(mask, depth_data, np.nan)
    else:
        masked_data = depth_data

    if use_mean:
        profile = np.nanmean(masked_data, axis=1)
    else:
        profile = np.nanmedian(masked_data, axis=1)

    return profile


# =============================================================================
# Main Preprocessing Pipeline (Steps 2 + 3)
# =============================================================================


def preprocess_data(
    depth_data: NDArray[np.floating],
    xdim: float,
    ydim: float | None = None,
    mark_type: str | None = None,
    mask: NDArray[np.bool_] | None = None,
    cutoff_hi: float = 2000e-6,
    cutoff_lo: float = 250e-6,
    cut_borders_after_smoothing: bool = True,
    use_mean: bool = True,
    angle_accuracy: float = 0.1,
    max_iter: int = 25,
    extra_sub_samp: int = 1,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.bool_] | None,
    float,
]:
    """
    Complete preprocessing pipeline for striated marks.

    Implements Steps 2 and 3 of PreprocessData.m:

    **Step 2: Form and noise removal**
        - Highpass filter to remove large-scale shape (curvature, tilt)
        - Lowpass filter to remove high-frequency noise

    **Step 3: Fine rotation and profile extraction**
        - Iteratively detect striation direction via gradient analysis
        - Rotate data to align striations horizontally
        - Extract mean or median profile

    :param depth_data: 2D depth/height data array.
    :param xdim: X-dimension pixel spacing in meters.
    :param ydim: Y-dimension pixel spacing in meters (optional, for resampling).
    :param mark_type: Mark type string (optional, for resampling).
    :param mask: Boolean mask array (True = valid data).
    :param cutoff_hi: Cutoff wavelength for shape removal (default 2000e-6 m).
    :param cutoff_lo: Cutoff wavelength for noise removal (default 250e-6 m).
    :param cut_borders_after_smoothing: If True, crop filter edge artifacts.
    :param use_mean: If True, use mean for profile; if False, use median.
    :param angle_accuracy: Target angle accuracy in degrees (default 0.1).
    :param max_iter: Maximum iterations for fine alignment.
    :param extra_sub_samp: Additional subsampling factor for gradient detection.

    :returns: Tuple of (aligned_data, profile, mask, total_angle).
    """
    # -------------------------------------------------------------------------
    # Step 2: Form and noise removal
    # -------------------------------------------------------------------------
    data_filtered, mask_filtered = apply_shape_noise_removal(
        depth_data=depth_data,
        xdim=xdim,
        cutoff_hi=cutoff_hi,
        mask=mask,
        cut_borders_after_smoothing=cut_borders_after_smoothing,
        cutoff_lo=cutoff_lo,
    )

    # -------------------------------------------------------------------------
    # Step 3: Fine rotation and profile extraction
    # -------------------------------------------------------------------------
    # Only perform fine alignment if data has more than 1 column
    if data_filtered.shape[1] > 1:
        data_aligned, mask_aligned, total_angle = fine_align_bullet_marks(
            depth_data=data_filtered,
            xdim=xdim,
            ydim=ydim,
            mark_type=mark_type,
            mask=mask_filtered,
            angle_accuracy=angle_accuracy,
            cut_y_after_shift=cut_borders_after_smoothing,
            max_iter=max_iter,
            extra_sub_samp=extra_sub_samp,
        )
    else:
        # Line profile - no alignment needed
        data_aligned = data_filtered
        mask_aligned = mask_filtered
        total_angle = 0.0

    # Extract profile
    profile = extract_profile(data_aligned, mask=mask_aligned, use_mean=use_mean)

    return data_aligned, profile, mask_aligned, total_angle
