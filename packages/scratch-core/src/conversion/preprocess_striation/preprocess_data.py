"""Preprocessing pipeline for striated tool and bullet marks.

This module implements the PreprocessData pipeline with the following steps:
- Form and noise removal (shape removal via highpass, noise removal via lowpass)
- Fine rotation to align striations horizontally + profile extraction
"""

from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType
from conversion.filter import cutoff_to_gaussian_sigma
from conversion.mask import _determine_bounding_box
from conversion.resample import resample_scan_image_and_mask
from conversion.preprocess_striation.preprocess_data_filter import (
    apply_gaussian_filter_1d,
)


def apply_shape_noise_removal(
    scan_image: ScanImage,
    mask: MaskArray | None = None,
    lowpass_cutoff: float = 5e-6,
    highpass_cutoff: float = 2.5e-4,
) -> tuple[NDArray[np.floating], MaskArray]:
    """
    Apply large-scale shape and noise removal to isolate striation features.

    The function has the following steps:

    **Calculate sigma and check data size**
        Convert the cutoff wavelength to Gaussian sigma. If the data is
        too short (2*sigma > 20% of height), disable border cutting to
        preserve data.

    **Shape removal**
        Use apply_gaussian_filter_1d with is_high_pass=True to remove
        large-scale shape (curvature, tilt, waviness).

    **Noise removal**
        Apply apply_gaussian_filter_1d wit  h is_high_pass=False (lowpass)
        to remove high-frequency noise while preserving striation features.


    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param mask: Boolean mask array (True = valid data).
    :param lowpass_cutoff: Low-frequency cutoff wavelength in meters (m) for noise removal.
    :param highpass_cutoff: High-frequency cutoff wavelength in meters (m) for shape removal.

    :returns: Tuple of (processed_data, mask).
    """
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(scan_image.data.shape, dtype=bool)

    # Calculate Gaussian sigma from cutoff wavelength
    sigma = cutoff_to_gaussian_sigma(highpass_cutoff, scan_image.scale_x)

    # Only crop borders if total removed (2*sigma for top+bottom) is ≤20% of height.
    # This preserves at least 80% of the data while removing edge artifacts.
    cut_borders = (2 * sigma) <= (scan_image.height * 0.2)

    # Shape removal (highpass filter)
    data_high_pass, mask_high_pass = apply_gaussian_filter_1d(
        scan_image=scan_image,
        cutoff=highpass_cutoff,
        is_high_pass=True,
        cut_borders_after_smoothing=cut_borders,
        mask=mask,
    )

    # Create intermediate ScanImage for noise removal
    intermediate_scan_image = scan_image.model_copy(update={"data": data_high_pass})

    # Noise removal (lowpass filter)
    data_no_noise, mask_no_noise = apply_gaussian_filter_1d(
        scan_image=intermediate_scan_image,
        cutoff=lowpass_cutoff,
        is_high_pass=False,
        cut_borders_after_smoothing=cut_borders,
        mask=mask_high_pass,
    )

    return data_no_noise, mask_no_noise


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


def _rotate_data_by_shifting_profiles(
    depth_data: NDArray[np.floating],
    angle_rad: float,
    cut_y_after_shift: bool = True,
) -> NDArray[np.floating]:
    """
    Rotate depth data by shifting each column (profile) vertically.

    This implements a shear-based rotation that preserves striation features.
    Instead of true 2D rotation (which would require interpolation in both
    dimensions), each column is shifted up or down by an amount proportional
    to its x-position. This creates a rotation effect while keeping each
    vertical profile intact.

    Example for a 5° (0.087 rad) rotation on a 100-pixel wide image:
        - Left edge (col 0): shifts up by ~4.4 pixels
        - Center (col 50): no shift
        - Right edge (col 99): shifts down by ~4.4 pixels

    :param depth_data: 2D depth data array (rows x cols).
    :param angle_rad: Rotation angle in radians (positive = clockwise).
        Expected range: ±0.175 rad (±10°). Angles < 0.00175 rad (0.1°) are skipped.
    :param cut_y_after_shift: If True, crop NaN borders introduced by shifting.
    :returns: Rotated depth data.
    """
    # Skip rotation for angles smaller than ~0.1° (0.00175 rad)
    if abs(angle_rad) <= 0.00175:
        return depth_data.copy()

    height, width = depth_data.shape

    # Calculate total vertical shift: tan(angle) * width
    # For small angles, tan(angle) ≈ angle, so shift ≈ angle_rad * width
    total_shift = np.tan(angle_rad) * width

    # Calculate shift for each column, centered around the middle column (pivot point).
    # Left edge shifts up by +total_shift/2, right edge shifts down by -total_shift/2.
    shift_y = np.linspace(total_shift / 2, -total_shift / 2, width)

    # Number of padding rows needed
    num_shift_rows = int(np.ceil(abs(total_shift) / 2))
    padding = num_shift_rows + 2

    # Pad data with NaN
    padded_height = height + 2 * padding
    padded_data = np.full((padded_height, width), np.nan)
    padded_data[padding : padding + height, :] = depth_data

    # Create coordinate grids for vectorized interpolation
    row_idx = np.arange(padded_height)
    col_idx = np.arange(width)

    # Source row for each output position: source_row = output_row - shift_y[col]
    # Broadcasting: (padded_height, 1) - (1, width) -> (padded_height, width)
    source_rows = row_idx[:, np.newaxis] - shift_y[np.newaxis, :]
    source_cols = np.broadcast_to(col_idx, (padded_height, width)).astype(float)

    # Stack coordinates for map_coordinates: shape (2, padded_height, width)
    coords = np.array([source_rows, source_cols])

    # Apply vectorized interpolation (order=1 for linear)
    output = map_coordinates(padded_data, coords, order=1, mode="constant", cval=np.nan)

    # Crop borders if requested
    if cut_y_after_shift:
        num_nan_rows = int(np.ceil(max(abs(shift_y[0]), abs(shift_y[-1])))) + 1
        crop_start = padding + num_nan_rows
        crop_end = padded_height - padding - num_nan_rows
        output = output[crop_start:crop_end, :]

    return output


def _rotate_image_grad_vector(
    depth_data: NDArray[np.floating],
    scale_x: float,
    mask: MaskArray | None = None,
    extra_sub_samp: int = 1,
) -> float:
    """
    Determine striation direction using gradient analysis.

    :param depth_data: 2D depth data array.
    :param scale_x: Pixel spacing in meters.
    :param mask: Optional boolean mask (True = valid).
    :param extra_sub_samp: Additional subsampling factor.
    :returns: Detected rotation angle in degrees.
    """
    # Determine subsampling factor
    if scale_x < 1e-6:
        sub_samp = ceil(1e-6 / scale_x) * extra_sub_samp
    else:
        sub_samp = 1 * extra_sub_samp

    # Determine sigma for smoothing (minimum of 3)
    sigma = max(3, round(1.75e-5 / scale_x / sub_samp))

    # Subsample data
    if sub_samp > 1 and depth_data.shape[1] // sub_samp >= 2:
        data_subsampled = depth_data[:, ::sub_samp]
        if mask is not None:
            mask_subsampled = mask[:, ::sub_samp]
        else:
            mask_subsampled = None
    else:
        data_subsampled = depth_data
        mask_subsampled = mask

    # Smooth data
    smoothed = _smooth_2d(data_subsampled, (sigma, sigma))

    # Calculate gradient
    fy, fx = np.gradient(smoothed)

    # Calculate total gradient magnitude
    grad_tmp = np.abs(fx) + np.abs(fy)

    # Create gradient threshold mask
    grad_threshold = 1.5 * np.nanmedian(grad_tmp)
    grad_mask = grad_tmp > grad_threshold

    # Combine with input mask if provided
    if mask_subsampled is not None:
        grad_mask = grad_mask & (mask_subsampled > 0.5)

    # Compute striation tilt angle from gradient direction.
    # For horizontal striations, gradients point vertically (fx=0, fy≠0).
    # Tilted striations produce a horizontal gradient component: sin(θ) = fx/|grad|.
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_norm = fx / grad_tmp

    # Ensure consistent sign by aligning with fy direction
    fx_norm = fx_norm * np.sign(fy)

    # Convert normalized fx to tilt angle in degrees via arcsin
    fx_flat = fx_norm.flatten()
    mask_flat = grad_mask.flatten()
    angles = np.degrees(np.arcsin(np.clip(fx_flat[mask_flat], -1, 1)))

    # Filter outliers: keep only angles within ±10° (expected range for fine alignment)
    angles = angles[np.abs(angles) < 10]

    if len(angles) == 0:
        return np.nan

    return float(np.median(angles))


def fine_align_bullet_marks(
    scan_image: ScanImage,
    mark_type: MarkType | None = None,
    mask: MaskArray | None = None,
    angle_accuracy: float = 0.1,
    cut_y_after_shift: bool = True,
    max_iter: int = 25,
    extra_sub_samp: int = 1,
) -> tuple[NDArray[np.floating], MaskArray | None, float]:
    """
    Fine alignment of striated marks by iteratively detecting striation direction.

    Iteratively determines the direction of striation marks and rotates the
    depth data so that striations are horizontal.

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param mark_type: Mark type enum value (optional, for resampling).
    :param mask: Optional boolean mask (True = valid).
    :param angle_accuracy: Target angle accuracy in degrees (default 0.1).
    :param cut_y_after_shift: If True, crop borders after shifting.
    :param max_iter: Maximum number of iterations.
    :param extra_sub_samp: Additional subsampling factor for gradient detection.
    :returns: Tuple of (aligned_data, aligned_mask, total_angle).
    """
    # Extract data and pixel spacing from ScanImage
    depth_data = scan_image.data
    scale_x = scan_image.scale_x

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
            scale_x,
            mask=mask_tmp,
            extra_sub_samp=extra_sub_samp,
        )

        if not np.isnan(a):
            a_tot = a_tot + a
            # Convert accumulated angle from degrees to radians for rotation
            a_tot_rad = np.radians(a_tot)
            data_tmp = _rotate_data_by_shifting_profiles(
                depth_data, a_tot_rad, cut_y_after_shift
            )

            if mask is not None:
                mask_float = mask.astype(float)
                mask_rotated = _rotate_data_by_shifting_profiles(
                    mask_float, a_tot_rad, cut_y_after_shift
                )
                mask_tmp = mask_rotated > 0.5
                y_slice, x_slice = _determine_bounding_box(mask_tmp)
                data_tmp = data_tmp[y_slice, x_slice]
                mask_tmp = mask_tmp[y_slice, x_slice]

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
        # Convert final angle from degrees to radians for rotation
        a_tot_rad = np.radians(a_tot)
        result_data = _rotate_data_by_shifting_profiles(
            depth_data, a_tot_rad, cut_y_after_shift
        )

        if mask is not None:
            mask_float = mask.astype(float)
            mask_rotated = _rotate_data_by_shifting_profiles(
                mask_float, a_tot_rad, cut_y_after_shift
            )
            result_mask = mask_rotated > 0.5
            y_slice, x_slice = _determine_bounding_box(result_mask)
            result_data = result_data[y_slice, x_slice]
            result_mask = result_mask[y_slice, x_slice]

            if mark_type is not None:
                temp_scan = scan_image.model_copy(update={"data": result_data})
                resampled_scan, result_mask = resample_scan_image_and_mask(
                    temp_scan,
                    result_mask,
                    target_scale=mark_type.scale,
                    only_downsample=True,
                )
                result_data = resampled_scan.data
        else:
            result_mask = None
            if mark_type is not None:
                temp_scan = scan_image.model_copy(update={"data": result_data})
                resampled_scan, _ = resample_scan_image_and_mask(
                    temp_scan, target_scale=mark_type.scale, only_downsample=True
                )
                result_data = resampled_scan.data

    return result_data, result_mask, a_tot


def extract_profile(
    depth_data: NDArray[np.floating],
    mask: MaskArray | None = None,
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


def preprocess_data(
    scan_image: ScanImage,
    mark_type: MarkType | None = None,
    mask: MaskArray | None = None,
    cutoff_hi: float = 2000e-6,
    cutoff_lo: float = 250e-6,
    cut_borders_after_smoothing: bool = True,
    use_mean: bool = True,
    angle_accuracy: float = 0.1,
    max_iter: int = 25,
    extra_sub_samp: int = 1,
    shape_noise_removal: bool = True,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    MaskArray | None,
    float,
]:
    """
    Complete preprocess_striations pipeline for striated marks, it performs two
    preprocessing steps:

    **Form and noise removal** (optional, controlled by shape_noise_removal)
        - Highpass filter to remove large-scale shape (curvature, tilt)
        - Lowpass filter to remove high-frequency noise

    **Fine rotation and profile extraction**
        - Iteratively detect striation direction via gradient analysis
        - Rotate data to align striations horizontally
        - Extract mean or median profile

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param mark_type: Mark type enum value (optional, for resampling).
    :param mask: Boolean mask array (True = valid data).
    :param cutoff_hi: Cutoff wavelength for shape removal (default 2000e-6 m).
    :param cutoff_lo: Cutoff wavelength for noise removal (default 250e-6 m).
    :param cut_borders_after_smoothing: If True, crop filter edge artifacts.
    :param use_mean: If True, use mean for profile; if False, use median.
    :param angle_accuracy: Target angle accuracy in degrees (default 0.1).
    :param max_iter: Maximum iterations for fine alignment.
    :param extra_sub_samp: Additional subsampling factor for gradient detection.
    :param shape_noise_removal: If True, apply shape and noise removal filters.

    :returns: Tuple of (aligned_data, profile, mask, total_angle).
    """
    if shape_noise_removal:
        data_filtered, mask_filtered = apply_shape_noise_removal(
            scan_image=scan_image,
            highpass_cutoff=cutoff_hi,
            mask=mask,
            lowpass_cutoff=cutoff_lo,
        )
    else:
        # Skip filtering - use original data
        data_filtered = scan_image.data.copy()
        mask_filtered = mask

    # Only perform fine alignment if data has more than 1 column
    if data_filtered.shape[1] > 1:
        # Create new ScanImage with filtered data
        filtered_scan_image = scan_image.model_copy(update={"data": data_filtered})
        data_aligned, mask_aligned, total_angle = fine_align_bullet_marks(
            scan_image=filtered_scan_image,
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
