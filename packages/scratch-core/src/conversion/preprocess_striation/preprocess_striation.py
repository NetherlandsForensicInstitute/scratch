"""Preprocessing pipeline for striated tool and bullet marks.

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
    apply_gaussian_filter_1d,
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
    Apply large-scale shape and noise removal to isolate striation features.
    This is basically a band-pass filter.

    The function has the following steps:

    **Calculate sigma and check data size**
        Convert the cutoff wavelength to Gaussian sigma. If the data is
        too short (2*sigma > 20% of height), disable border cutting to
        preserve data.

    **Shape removal**
        Use apply_gaussian_filter_1d with is_high_pass=True to remove
        large-scale shape (curvature, tilt, waviness).

    **Noise removal**
        Apply apply_gaussian_filter_1d with is_high_pass=False (lowpass)
        to remove high-frequency noise while preserving striation features.

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


def _shear_data_by_shifting_profiles(
    depth_data: NDArray[np.floating],
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
    extra_sub_samp: int = 1,
) -> float:
    """
    Determine striation direction using gradient analysis.

    :param depth_data: 2D depth data array.
    :param scale_x: Pixel spacing in meters.
    :param mask: Optional boolean mask (True = valid).
    :param extra_sub_samp: Additional subsampling factor on top of the automatic subsampling for faster calculation,
    but lower precision.
    :returns: Detected rotation angle in degrees.
    """
    # Determine subsampling factor
    sub_samp = extra_sub_samp
    if scale_x < 1e-6:
        sub_samp = round(1e-6 / scale_x) * extra_sub_samp

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
    extra_sub_samp: int,
) -> float:
    """
    Iteratively find the alignment angle for striation marks.

    :param depth_data: 2D depth data array.
    :param mask: Optional boolean mask.
    :param scale_x: Pixel spacing in meters.
    :param angle_accuracy: Target angle accuracy in degrees.
    :param cut_y_after_shift: If True, crop borders after shifting.
    :param max_iter: Maximum number of iterations.
    :param extra_sub_samp: Additional subsampling factor.
    :returns: Total alignment angle in degrees (0.0 if max_iter reached).
    """
    data_tmp = depth_data.copy()
    mask_tmp = mask.copy() if mask is not None else None

    a_tot = 0.0
    a = -45.0
    a_last = 0.0
    iteration = 1

    while abs(a) > angle_accuracy and iteration < max_iter:
        a = _rotate_image_grad_vector(
            data_tmp, scale_x, mask=mask_tmp, extra_sub_samp=extra_sub_samp
        )

        if np.isnan(a):
            a = 0.05
        else:
            a_tot = a_tot + a
            a_tot_rad = np.radians(a_tot)
            data_tmp, mask_tmp = _rotate_data_and_mask(
                depth_data, mask, a_tot_rad, cut_y_after_shift
            )

            if a == a_last:
                iteration = max_iter - 1
            else:
                a_last = a

        iteration += 1

    if iteration >= max_iter:
        return 0.0
    return a_tot


def _rotate_data_and_mask(
    depth_data: NDArray[np.floating],
    mask: MaskArray | None,
    angle_rad: float,
    cut_y_after_shift: bool,
) -> tuple[NDArray[np.floating], MaskArray | None]:
    """
    Rotate data and optionally mask, cropping to bounding box if mask provided.

    :param depth_data: 2D depth data array.
    :param mask: Optional boolean mask array.
    :param angle_rad: Rotation angle in radians.
    :param cut_y_after_shift: If True, crop NaN borders after shifting.
    :returns: Tuple of (rotated_data, rotated_mask).
    """
    if mask is not None:
        data_rotated = _shear_data_by_shifting_profiles(
            depth_data, angle_rad, cut_y_after_shift
        )
        mask_rotated = _shear_data_by_shifting_profiles(
            mask.astype(float), angle_rad, cut_y_after_shift
        )
        mask_binary = mask_rotated == 1
        y_slice, x_slice = _determine_bounding_box(mask_binary)
        return data_rotated[y_slice, x_slice], mask_binary[y_slice, x_slice]

    return _shear_data_by_shifting_profiles(
        depth_data, angle_rad, cut_y_after_shift
    ), None


def fine_align_bullet_marks(
    scan_image: ScanImage,
    mark_type: MarkType | None = None,
    mask: MaskArray | None = None,
    angle_accuracy: float = 0.1,
    cut_y_after_shift: bool = True,
    max_iter: int = 25,
    extra_sub_samp: int = 1,
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
    :param extra_sub_samp: Additional subsampling factor for gradient detection.
    :returns: Tuple of (aligned_scan_image, aligned_mask, total_angle_degrees).
    """
    depth_data = scan_image.data
    scale_x = scan_image.scale_x
    scale_y = scan_image.scale_y

    # Find alignment angle iteratively
    a_tot = _find_alignment_angle(
        depth_data,
        mask,
        scale_x,
        angle_accuracy,
        cut_y_after_shift,
        max_iter,
        extra_sub_samp,
    )

    # Apply rotation and compute corrected scales
    result_data = depth_data.copy()
    result_mask = mask
    result_scale_x = scale_x
    result_scale_y = scale_y

    if not np.isclose(a_tot, 0.0, rtol=1e-09, atol=1e-09):
        a_tot_rad = np.radians(a_tot)
        cos_angle = np.cos(a_tot_rad)
        result_scale_x = scale_x * cos_angle
        result_scale_y = scale_y / cos_angle
        result_data, result_mask = _rotate_data_and_mask(
            depth_data, mask, a_tot_rad, cut_y_after_shift
        )

    # Create result ScanImage with corrected scales
    result_scan = ScanImage(
        data=np.asarray(result_data, dtype=np.float64),
        scale_x=result_scale_x,
        scale_y=result_scale_y,
    )

    # Resample to mark type target scale if specified
    if mark_type is not None:
        result_scan, result_mask = resample_scan_image_and_mask(
            result_scan,
            result_mask,
            target_scale=mark_type.scale,
            only_downsample=True,
        )

    return result_scan, result_mask, a_tot


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
    masked_data = depth_data
    if mask is not None:
        masked_data = np.where(mask, depth_data, np.nan)

    if use_mean:
        return np.nanmean(masked_data, axis=1)
    return np.nanmedian(masked_data, axis=1)


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

    # Set defaults for line profile case (no alignment needed)
    data_aligned = data_filtered
    mask_aligned = mask_filtered
    total_angle = 0.0

    if data_filtered.shape[1] > 1:
        filtered_scan_image = scan_image.model_copy(update={"data": data_filtered})
        aligned_scan, mask_aligned, total_angle = fine_align_bullet_marks(
            scan_image=filtered_scan_image,
            mark_type=mark_type,
            mask=mask_filtered,
            angle_accuracy=params.angle_accuracy,
            cut_y_after_shift=params.cut_borders_after_smoothing,
            max_iter=params.max_iter,
            extra_sub_samp=params.extra_sub_samp,
        )
        data_aligned = aligned_scan.data

    # Propagate NaN to adjacent pixels to match MATLAB's asymmetric NaN handling
    data_aligned = _propagate_nan(data_aligned)

    # Extract profile
    profile = extract_profile(data_aligned, mask=mask_aligned, use_mean=params.use_mean)

    return data_aligned, profile, mask_aligned, total_angle
