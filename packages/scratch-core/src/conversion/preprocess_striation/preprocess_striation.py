"""
Preprocessing pipeline for striated tool and bullet marks.

This module implements the PreprocessData pipeline with the following steps:
- Form and noise removal (shape removal via highpass, noise removal via lowpass)
- Fine rotation to align striations horizontally + profile extraction
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType
from conversion.filter import (
    cutoff_to_gaussian_sigma,
    gaussian_sigma_to_cutoff,
    apply_striation_preserving_filter_1d,
    apply_gaussian_regression_filter,
)

from conversion.mask import _determine_bounding_box
from conversion.resample import resample_scan_image_and_mask, resample_image_array
from conversion.preprocess_striation.parameters import PreprocessingStriationParams


def apply_shape_noise_removal(
    scan_image: ScanImage,
    mask: MaskArray | None = None,
    lowpass_cutoff: float = 5e-6,
    highpass_cutoff: float = 2.5e-4,
) -> tuple[NDArray[np.floating], MaskArray]:
    """
    Apply a band-pass filter to isolate striation features by filtering out large-scale shapes and small-scale noise.

    The function has the following steps:

    - Calculate sigma and check data size
    - Shape removal (curvature, tilt, waviness)
    - Noise removal

    Note: we remove shape then noise by subsequently applying two Gaussian high_pass filters (first σ_low, then σ_high):
        gaussian(image - gaussian(image, σ_low), σ_high).
        Since noise removal is additive, this is the same as:
       = gaussian(image, σ_high) - gaussian(gaussian(image, σ_low), σ_high),
       which is the same as subtracting two Gaussian high_pass filters with different sigma:
       = gaussian(image, σ_high) - gaussian(image, √(σ_low² + σ_high²))

       Normally, such 'intermediate band' selection is performed by defining σ_high and σ_low, and then using:
       gaussian(image, σ_high) - gaussian(image, σ_low), which is called the Difference of Gaussians (DOG) procedure.
       So in our case, we could substitute in DOG: var_lowest_sigma = σ_low² + σ_high² and then apply DOG.
       So our procedure is identical to DOG but using different definition of σ_low.


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
    data_high_pass, mask_high_pass = apply_striation_preserving_filter_1d(
        scan_image=scan_image,
        cutoff=highpass_cutoff,
        is_high_pass=True,
        cut_borders_after_smoothing=cut_borders,
        mask=mask,
    )

    # Create intermediate ScanImage for noise removal
    intermediate_scan_image = scan_image.model_copy(update={"data": data_high_pass})

    # Noise removal (lowpass filter)
    data_no_noise, mask_no_noise = apply_striation_preserving_filter_1d(
        scan_image=intermediate_scan_image,
        cutoff=lowpass_cutoff,
        is_high_pass=False,
        cut_borders_after_smoothing=cut_borders,
        mask=mask_high_pass,
    )

    return data_no_noise, mask_no_noise


def _shear_data_by_shifting_profiles(
    depth_data: NDArray[np.floating] | MaskArray,
    angle_rad: float,
    cut_y_after_shift: bool = True,
) -> NDArray[np.floating]:
    """
    Shear depth data by shifting each column (profile) vertically.

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
    subsampling_factor: int = 1,
) -> float:
    """
    Determine striation direction using gradient analysis.

    For horizontal striations, gradients point vertically (fx=0, fy≠0).
    Tilted striations produce a horizontal gradient component, from which
    the tilt angle is computed as: θ = arcsin(fx / |gradient|).

    :param depth_data: 2D depth data array.
    :param scale_x: Pixel spacing in meters.
    :param mask: Optional boolean mask (True = valid).
    :param subsampling_factor: Additional subsampling factor on top of the automatic subsampling for faster calculation,
    but lower precision.
    :returns: Detected rotation angle in degrees (positive = clockwise).
    """
    # Determine subsampling factor
    sub_samp = subsampling_factor
    if scale_x < 1e-6:
        sub_samp = round(1e-6 / scale_x) * subsampling_factor

    # Resample data (only columns, matching MATLAB's resample function)
    data_subsampled = depth_data
    mask_subsampled = mask

    if sub_samp > 1 and depth_data.shape[1] // sub_samp >= 2:
        data_subsampled = resample_image_array(depth_data, factors=(sub_samp, 1))
        if mask is not None:
            mask_subsampled = resample_image_array(mask, factors=(sub_samp, 1)) > 0.5

    # Smooth data using Gaussian filter (MATLAB-equivalent)
    # Original: sigma = max(3, round(1.75e-5 / effective_pixel_size)) in pixels
    effective_pixel_size = scale_x * sub_samp
    sigma_pixels = max(3, round(1.75e-5 / effective_pixel_size))
    smoothing_cutoff = gaussian_sigma_to_cutoff(sigma_pixels, effective_pixel_size)

    smoothed = apply_gaussian_regression_filter(
        data_subsampled,
        cutoff_length=smoothing_cutoff,
        pixel_size=(effective_pixel_size, effective_pixel_size),
        regression_order=0,
        nan_out=True,
    )

    # Calculate gradient
    fy, fx = np.gradient(smoothed)

    # Calculate total gradient (L1 norm)
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


def _find_alignment_angle(
    depth_data: NDArray[np.floating],
    mask: MaskArray | None,
    scale_x: float,
    angle_accuracy: float,
    cut_y_after_shift: bool,
    max_iter: int,
    subsampling_factor: int,
) -> float:
    """
    Iteratively find the alignment angle for striation marks.

    The alignment angle is the rotation needed to make striations horizontal.
    Positive angle means clockwise rotation.

    :param depth_data: 2D depth data array.
    :param mask: Optional boolean mask.
    :param scale_x: Pixel spacing in meters.
    :param angle_accuracy: Target angle accuracy in degrees.
    :param cut_y_after_shift: If True, crop borders after shifting.
    :param max_iter: Maximum number of iterations.
    :param subsampling_factor: Additional subsampling factor.
    :returns: Total alignment angle in degrees (0.0 if max_iter reached).
    """
    data_tmp = depth_data
    mask_tmp = mask

    total_angle = 0.0
    current_angle = -45.0
    previous_angle = 0.0
    iteration = 1

    # Iteratively refine the alignment angle until striations are horizontal enough.
    # Each iteration detects the remaining misalignment, accumulates it into total_angle,
    # then re-rotates the original data by total_angle and re-measures.
    # Stopping criteria:
    #   - Converged: abs(current_angle) <= angle_accuracy
    #   - Max iterations reached (returns 0.0)
    #   - Stuck: current_angle == previous_angle (forces early exit)
    while abs(current_angle) > angle_accuracy and iteration < max_iter:
        current_angle = _rotate_image_grad_vector(
            data_tmp, scale_x, mask=mask_tmp, subsampling_factor=subsampling_factor
        )

        if np.isnan(current_angle):
            current_angle = 0.05
        else:
            total_angle += current_angle
            total_angle_rad = np.radians(total_angle)
            data_tmp, mask_tmp = _rotate_data_and_mask(
                depth_data, mask, total_angle_rad, cut_y_after_shift
            )

            if current_angle == previous_angle:
                iteration = max_iter - 1
            else:
                previous_angle = current_angle

        iteration += 1

    return total_angle if iteration < max_iter else 0.0


def _rotate_data_and_mask(
    depth_data: NDArray[np.floating],
    mask: MaskArray | None,
    angle_rad: float,
    crop_nan_borders: bool,
) -> tuple[NDArray[np.floating], MaskArray | None]:
    """
    Rotate data and optionally mask, cropping to bounding box if mask provided.

    :param depth_data: 2D depth data array.
    :param mask: Optional boolean mask array.
    :param angle_rad: Rotation angle in radians.
    :param crop_nan_borders: If True, crop NaN borders after shifting.
    :returns: Tuple of (rotated_data, rotated_mask).
    """
    if mask is not None:
        data_rotated = _shear_data_by_shifting_profiles(
            depth_data, angle_rad, crop_nan_borders
        )
        mask_rotated = (
            _shear_data_by_shifting_profiles(mask, angle_rad, crop_nan_borders) > 0.5
        )
        y_slice, x_slice = _determine_bounding_box(mask_rotated)
        return data_rotated[y_slice, x_slice], mask_rotated[y_slice, x_slice]

    return _shear_data_by_shifting_profiles(
        depth_data, angle_rad, crop_nan_borders
    ), None


def _create_shear_corrected_scan_image(
    data: NDArray[np.floating],
    scale_x: float,
    scale_y: float,
    angle_degrees: float,
) -> ScanImage:
    """
    Create a ScanImage with scales corrected for shear-based rotation.

    Shearing compresses the x-axis by cos(angle) and expands the y-axis by 1/cos(angle).

    :param data: 2D depth data array.
    :param scale_x: Original x pixel spacing in meters.
    :param scale_y: Original y pixel spacing in meters.
    :param angle_degrees: Applied rotation angle in degrees.
    :returns: ScanImage with corrected scales.
    """
    if not np.isclose(angle_degrees, 0.0, rtol=1e-09, atol=1e-09):
        cos_angle = np.cos(np.radians(angle_degrees))
        scale_x = scale_x * cos_angle
        scale_y = scale_y / cos_angle

    return ScanImage(
        data=np.asarray(data, dtype=np.float64),
        scale_x=scale_x,
        scale_y=scale_y,
    )


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

    Iteratively determines the direction of striation marks and rotates the
    depth data so that striations are horizontal. Scales are corrected for
    the shear-based rotation: scale_x is compressed by cos(angle) and
    scale_y is expanded by 1/cos(angle).

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
        scan_image.data,
        mask,
        scan_image.scale_x,
        angle_accuracy,
        cut_y_after_shift,
        max_iter,
        subsampling_factor,
    )

    # Apply rotation and crop to mask bounding box
    if not np.isclose(total_angle, 0.0, rtol=1e-09, atol=1e-09):
        total_angle_rad = np.radians(total_angle)
        result_data, result_mask = _rotate_data_and_mask(
            scan_image.data, mask, total_angle_rad, cut_y_after_shift
        )
    elif mask is not None:
        # No rotation needed, but still crop to mask bounding box
        y_slice, x_slice = _determine_bounding_box(mask)
        result_data = scan_image.data[y_slice, x_slice]
        result_mask = mask[y_slice, x_slice]
    else:
        result_data = scan_image.data
        result_mask = mask

    result_scan = _create_shear_corrected_scan_image(
        result_data, scan_image.scale_x, scan_image.scale_y, total_angle
    )

    # Resample to mark type target scale if specified
    if mark_type is not None:
        result_scan, result_mask = resample_scan_image_and_mask(
            result_scan,
            result_mask,
            target_scale=mark_type.scale,
            only_downsample=True,
        )

    return result_scan, result_mask, total_angle


def _propagate_nan(data: NDArray[np.floating]) -> NDArray[np.floating]:
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


def preprocess_data(
    scan_image: ScanImage,
    mark_type: MarkType | None = None,
    mask: MaskArray | None = None,
    params: PreprocessingStriationParams | None = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    MaskArray | None,
    float,
]:
    """
    Complete preprocess_striations pipeline for striated marks, it performs two
    preprocessing steps:

    **Form and noise removal**
        - Highpass filter to remove large-scale shape (curvature, tilt)
        - Lowpass filter to remove high-frequency noise

    **Fine rotation and profile extraction**
        - Iteratively detect striation direction via gradient analysis
        - Rotate data to align striations horizontally
        - Extract mean or median profile

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param mark_type: Mark type enum value (optional, for resampling).
    :param mask: Boolean mask array (True = valid data).
    :param params: Preprocessing parameters. If None, uses default values.

    :returns: Tuple of (aligned_data, profile, mask, total_angle).
    """
    if params is None:
        params = PreprocessingStriationParams()

    data_filtered, mask_filtered = apply_shape_noise_removal(
        scan_image=scan_image,
        highpass_cutoff=params.cutoff_hi,
        mask=mask,
        lowpass_cutoff=params.cutoff_lo,
    )

    if data_filtered.shape[1] > 1:
        filtered_scan_image = scan_image.model_copy(update={"data": data_filtered})
        aligned_scan, mask_aligned, total_angle = fine_align_bullet_marks(
            scan_image=filtered_scan_image,
            mark_type=mark_type,
            mask=mask_filtered,
            angle_accuracy=params.angle_accuracy,
            cut_y_after_shift=params.cut_borders_after_smoothing,
            max_iter=params.max_iter,
            subsampling_factor=params.subsampling_factor,
        )
        data_aligned = aligned_scan.data
    else:
        # Line profile case (no alignment needed)
        data_aligned = data_filtered
        mask_aligned = mask_filtered
        total_angle = 0.0

    # Propagate NaN to adjacent pixels to match MATLAB's asymmetric NaN handling
    data_aligned = _propagate_nan(data_aligned)

    # Extract profile: apply mask and compute mean/median along rows
    if mask_aligned is not None:
        data_aligned = np.where(mask_aligned, data_aligned, np.nan)
    profile = (
        np.nanmean(data_aligned, axis=1)
        if params.use_mean
        else np.nanmedian(data_aligned, axis=1)
    )

    return data_aligned, profile, mask_aligned, total_angle
