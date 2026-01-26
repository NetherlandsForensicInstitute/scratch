"""
Main preprocessing pipeline for striated tool and bullet marks.

This module provides the high-level entry points for striation preprocessing:
- Form and noise removal (shape removal via highpass, noise removal via lowpass)
- Fine rotation to align striations horizontally and profile extraction
"""

from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.filter import (
    cutoff_to_gaussian_sigma,
    apply_striation_preserving_filter_1d,
)
from conversion.preprocess_striation.parameters import PreprocessingStriationParams
from conversion.preprocess_striation.alignment import fine_align_bullet_marks
from conversion.preprocess_striation.shear import propagate_nan


def preprocess_striation_mark(
    mark: Mark,
    params: PreprocessingStriationParams = PreprocessingStriationParams(),
) -> tuple[Mark, Mark]:
    """
    Complete the preprocessing pipeline for striated marks.

    Performs two preprocessing steps:

    **Form and noise removal**
        - Highpass filter to remove large-scale shape (curvature, tilt)
        - Lowpass filter to remove high-frequency noise

    **Fine rotation and profile extraction**
        - Iteratively detect the striation direction via gradient analysis
        - Rotate data to align striations horizontally
        - Extract mean or median profile

    :param mark: Input Mark object containing scan_image and mark_type.
    :param params: Preprocessing parameters.

    :returns: Tuple of (aligned_mark, profile_mark).
        - aligned_mark: Mark with aligned striation data, mask and total_angle in meta_data.
        - profile_mark: Mark with extracted 1D profile (as 2D array with shape (N, 1)),
          mask and total_angle in meta_data.
    """
    scan_image = mark.scan_image
    mark_type = mark.mark_type

    data_filtered = apply_shape_noise_removal(
        scan_image=scan_image,
        highpass_cutoff=params.highpass_cutoff,
        lowpass_cutoff=params.lowpass_cutoff,
    )

    if data_filtered.shape[1] > 1:
        filtered_scan_image = scan_image.model_copy(update={"data": data_filtered})
        aligned_scan, mask_aligned, total_angle = fine_align_bullet_marks(
            scan_image=filtered_scan_image,
            mark_type=mark_type,
            angle_accuracy=params.angle_accuracy,
            cut_y_after_shift=params.cut_borders_after_smoothing,
            max_iter=params.max_iter,
            subsampling_factor=params.subsampling_factor,
        )
        data_aligned = aligned_scan.data
        scale_x = aligned_scan.scale_x
        scale_y = aligned_scan.scale_y
    else:
        # Line profile case (no alignment needed)
        data_aligned = data_filtered
        mask_aligned = None
        total_angle = 0.0
        scale_x = scan_image.scale_x
        scale_y = scan_image.scale_y

    # Propagate NaN to adjacent pixels to match MATLAB's asymmetric NaN handling
    data_aligned = propagate_nan(data_aligned)

    # Extract profile: apply mask and compute mean/median along rows
    if mask_aligned is not None:
        data_for_profile = np.where(mask_aligned, data_aligned, np.nan)
    else:
        data_for_profile = data_aligned

    profile = (
        np.nanmean(data_for_profile, axis=1)
        if params.use_mean
        else np.nanmedian(data_for_profile, axis=1)
    )

    # Build meta_data with mask and total_angle
    aligned_meta_data = {
        **mark.meta_data,
        **asdict(params),
        "total_angle": total_angle,
    }
    if mask_aligned is not None:
        aligned_meta_data["mask"] = mask_aligned.tolist()

    profile_meta_data = {
        **mark.meta_data,
        **asdict(params),
        "total_angle": total_angle,
    }

    # Create aligned mark
    aligned_mark = Mark(
        scan_image=ScanImage(
            data=np.asarray(data_aligned, dtype=np.float64),
            scale_x=scale_x,
            scale_y=scale_y,
        ),
        mark_type=mark_type,
        crop_type=mark.crop_type,
        meta_data=aligned_meta_data,
    )

    # Create a profile mark (profile as 2D array with shape (N, 1))
    profile_mark = Mark(
        scan_image=ScanImage(
            data=profile.reshape(-1, 1),
            scale_x=scale_x,
            scale_y=scale_y,
        ),
        mark_type=mark_type,
        crop_type=mark.crop_type,
        meta_data=profile_meta_data,
    )

    return aligned_mark, profile_mark


def apply_shape_noise_removal(
    scan_image: ScanImage,
    lowpass_cutoff: float = 5e-6,
    highpass_cutoff: float = 2.5e-4,
) -> NDArray[np.floating]:
    """
    Apply a band-pass filter to isolate striation features by filtering out large-scale shapes and small-scale noise.

    The function has the following steps:

    - Calculate sigma and check data size
    - Shape removal (curvature, tilt, waviness)
    - Noise removal

    Note: we remove shape then noise by subsequently applying two Gaussian highpass filters
    (first σ_low, then σ_high). This is equivalent to applying a Difference of Gaussians filter
    (https://en.wikipedia.org/wiki/Difference_of_Gaussians) with t_1 = σ_high and t_2 = √(σ_low² + σ_high²).


    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param lowpass_cutoff: Low-frequency cutoff wavelength in meters (m) for noise removal.
    :param highpass_cutoff: High-frequency cutoff wavelength in meters (m) for shape removal.

    :returns: Tuple of (processed_data, mask).
    """

    # Calculate Gaussian sigma from cutoff wavelength
    sigma = cutoff_to_gaussian_sigma(highpass_cutoff, scan_image.scale_x)

    # Only crop borders if total removed (2*sigma for top+bottom) is ≤20% of height.
    # This preserves at least 80% of the data while removing edge artifacts.
    cut_borders = (2 * sigma) <= (scan_image.height * 0.2)

    # Shape removal (highpass filter)
    data_high_pass = apply_striation_preserving_filter_1d(
        scan_image=scan_image,
        cutoff=highpass_cutoff,
        is_high_pass=True,
        cut_borders_after_smoothing=cut_borders,
    )

    # Create an intermediate ScanImage for noise removal
    intermediate_scan_image = scan_image.model_copy(update={"data": data_high_pass})

    # Noise removal (lowpass filter)
    data_no_noise = apply_striation_preserving_filter_1d(
        scan_image=intermediate_scan_image,
        cutoff=lowpass_cutoff,
        is_high_pass=False,
        cut_borders_after_smoothing=cut_borders,
    )

    return data_no_noise
