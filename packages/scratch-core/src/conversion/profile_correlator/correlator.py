"""Main entry point for profile correlation.

This module provides the primary interface for comparing striated mark profiles.
It uses a brute-force approach: try all possible shifts and scales between
profiles and select the one with maximum cross-correlation, subject to a
minimum overlap constraint.

All length and height measurements are in meters (SI units).

Comparison with Previous Implementation
----------------------------------------
This implementation replaces the complex multi-scale optimization approach with
a brute-force method that:

1. **Equalizes pixel scales** between profiles (same as before)
2. **Tries multiple scale factors** (e.g., 0.95, 0.97, ..., 1.03, 1.05)
3. **For each scale, tries all shifts** with at least min_overlap_distance overlap
4. **Computes correlation** at each shift position
5. **Selects maximum** correlation as the optimal alignment

Advantages:
- **Simpler**: ~300 lines vs 3000+ lines
- **More robust**: Guaranteed to find global maximum correlation
- **Easier to debug**: No complex multi-stage optimization
- **Deterministic**: Same inputs always produce same outputs

The output ComparisonResults is fully compatible with the previous implementation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    ComparisonResults,
    Profile,
)
from conversion.profile_correlator.transforms import equalize_pixel_scale


def _apply_scaling(
        data: NDArray[np.floating],
        scale_factor: float,
) -> NDArray[np.floating]:
    """
    Apply scaling transformation to a profile.

    Scale factor > 1.0 stretches the profile (samples later in data).
    Scale factor < 1.0 compresses the profile (samples earlier in data).

    :param data: Input 1D profile data.
    :param scale_factor: Scaling factor (1.0 = no scaling).
    :returns: Scaled profile with same length as input.
    """
    if scale_factor == 1.0:
        return data

    n = len(data)

    # Original sample indices (1-based to match MATLAB)
    x_orig = np.arange(1, n + 1, dtype=np.float64)

    # Create interpolator: data is at original positions
    interpolator = interp1d(
        x_orig,  # Data positions (original)
        data,  # Data values
        kind='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    # Sample at scaled positions
    # scale > 1.0: sample later (stretch)
    # scale < 1.0: sample earlier (compress)
    x_sample = x_orig * scale_factor

    return interpolator(x_sample)


def _compute_correlation(
        profile_ref: NDArray[np.floating],
        profile_comp: NDArray[np.floating],
) -> float:
    """
    Compute correlation between two profiles at a specific shift.

    Positive shift means profile_comp is shifted to the right relative to profile_ref.

    :param profile_ref: (Part of) reference profile (1D array).
    :param profile_comp: (Part of) comparison profile (1D array).
    :returns: correlation.
    """

    # Find valid (non-NaN) samples in both profiles
    valid_mask = ~(np.isnan(profile_ref) | np.isnan(profile_comp))

    if not np.any(valid_mask):
        return np.nan

    ref_valid = profile_ref[valid_mask]
    comp_valid = profile_comp[valid_mask]

    # Compute Pearson correlation
    ref_centered = ref_valid - np.mean(ref_valid)
    comp_centered = comp_valid - np.mean(comp_valid)

    numerator = np.dot(ref_centered, comp_centered)
    denominator = np.sqrt(
        np.dot(ref_centered, ref_centered) * np.dot(comp_centered, comp_centered)
    )

    if denominator == 0:
        return np.nan

    correlation = numerator / denominator

    return float(correlation)


def correlate_profiles(
        profile_ref: Profile,
        profile_comp: Profile,
        params: AlignmentParameters = AlignmentParameters(),
) -> ComparisonResults:
    """
    Compare two striated mark profiles and compute similarity metrics.

    This function performs profile alignment using a brute-force approach:
    it tries multiple scale factors and all possible shifts between the profiles
    (subject to a minimum overlap constraint) and selects the combination with
    maximum cross-correlation.

    The workflow is:

    1. **Equalize sampling distances**: If the two profiles have different pixel
       sizes, the higher-resolution one is resampled to match the lower resolution.

    2. **Try multiple scale factors**: Test scale factors in the range defined by
       params.max_scaling (e.g., ±5% means 0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05).

    3. **For each scale, determine shift range**: Calculate the range of valid shifts
       that maintain at least 200 micrometers of overlap.

    4. **Brute-force search**: Try all shift/scale combinations and compute
       cross-correlation at each position.

    5. **Select optimal alignment**: Choose the shift/scale with maximum correlation.

    6. **Compute metrics**: Calculate comprehensive comparison metrics including
       correlation coefficient, roughness parameters, and signature differences.

    All measurements are in meters (SI units).

    :param profile_ref: Reference profile to compare against.
    :param profile_comp: Compared profile to align to the reference.
    :param params: Alignment parameters. Used parameters:
        - partial_mark_threshold: Threshold for partial profile classification
        - use_mean: Use mean (True) or median (False) for multi-column profiles
        - max_scaling: Maximum scaling deviation (e.g., 0.05 for ±5%)
    :returns: ComparisonResults containing all computed metrics including:
        - correlation_coefficient: Pearson correlation after alignment
        - position_shift: Translation applied (m)
        - scale_factor: Scaling factor applied (e.g., 0.97, 1.0, 1.03)
        - overlap_length: Length of overlapping region (m)
        - overlap_ratio: Ratio of overlap to shorter profile
        - sa_ref, sq_ref: Roughness metrics for reference (m)
        - sa_comp, sq_comp: Roughness metrics for compared profile (m)
        - ds_combined: Combined signature difference

    Notes
    -----
    The function automatically handles:

    - Different pixel sizes between profiles
    - Different profile lengths
    - NaN values in the profile data
    - Multi-column profile data (averaged before comparison)

    The minimum overlap distance is fixed at 200 micrometers.

    Scale factors are tested at discrete intervals. By default, with max_scaling=0.05,
    the following scale factors are tried: [0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05].
    """
    # Minimum overlap distance in meters (200 micrometers)
    # For very short profiles, use a percentage instead
    min_overlap_distance = params.min_overlap_distance

    # Step 1: Equalize pixel scales
    profile_ref_eq, profile_comp_eq = equalize_pixel_scale(profile_ref, profile_comp)

    pixel_size = profile_ref_eq.pixel_size

    # Get 1D profiles (mean across columns if multi-column)
    ref_data = profile_ref_eq.mean_profile(use_mean=params.use_mean)
    comp_data_original = profile_comp_eq.mean_profile(use_mean=params.use_mean)

    len_ref = len(ref_data)
    len_comp_original = len(comp_data_original)


    # Step 2: Generate scale factors to try
    # Create discrete scale factors in the range [1-max_scaling, 1+max_scaling]
    # Use approximately 7-9 scale factors for good coverage
    num_scale_steps = 7
    scale_factors = np.linspace(
        1.0 - params.max_scaling,
        1.0 + params.max_scaling,
        num_scale_steps
    )

    # Step 3: Try all scale factors and allowed shifts
    best_correlation = -np.inf
    best_shift = 0
    best_scale = 1.0
    best_ref_overlap = None
    best_comp_overlap = None

    for scale in scale_factors:
        # Apply scaling to comparison profile
        comp_data_scaled = _apply_scaling(comp_data_original, scale)
        len_comp = len(comp_data_scaled)

        # Get boundaries overlap idx for reference_profile (a small amount larger than 200 mum):
        len_overlap = int(min_overlap_distance/pixel_size)
        if len_comp < len_overlap or len_ref < len_overlap:
            # One of profiles too small to give sufficient overlap. Do not calculate anything.
            continue
        else:
            # move large profile from left to right. Start with smaller profile on far right side.
            if len_ref >= len_comp:
                large = ref_data
                small = comp_data_scaled
            else:
                large = comp_data_scaled
                small = ref_data

            len_large = len(large)
            len_small = len(small)

            min_shift = -(len_small - len_overlap)
            max_shift = len_large - len_overlap

            for shift in range(min_shift, max_shift + 1):
                # Calculate overlap length for this shift. We shift large to the left (for positive shifts)
                shift = int(shift)
                overlap_len = max(min(
                    len_large - max(0, shift),  # How much of large is available
                    len_small - max(0, -shift)),  # How much of small is available
                    len_overlap  # Maximum we want
                )

                # Now extract exactly overlap_len samples from each
                idx_large_start = int(max(0, shift))
                partial_large = large[idx_large_start:idx_large_start + overlap_len]

                idx_small_start = int(max(0, -shift))
                partial_small = small[idx_small_start:idx_small_start + overlap_len]

                if len_ref >= len_comp:
                    partial_ref = partial_large
                    partial_comp = partial_small
                else:
                    partial_comp = partial_small
                    partial_ref = partial_large


                correlation = _compute_correlation(
                    partial_ref, partial_comp
                )

                if not np.isnan(correlation) and correlation > best_correlation:
                    best_correlation = correlation
                    best_shift = shift
                    best_scale = scale
                    best_ref_overlap = partial_ref
                    best_comp_overlap = partial_comp


    # Step 4: Compute metrics for the best alignment
    if best_ref_overlap is None or best_comp_overlap is None:
        # No valid alignment found - return empty results
        return ComparisonResults(
            is_profile_comparison=True,
            is_partial_profile=False,
            pixel_size_ref=pixel_size,
            pixel_size_comp=pixel_size,
            position_shift=np.nan,
            scale_factor=1.0,
            partial_start_position=np.nan,
            similarity_value=np.nan,
            overlap_length=np.nan,
            overlap_ratio=np.nan,
            correlation_coefficient=np.nan,
            sa_ref=np.nan,
            sq_ref=np.nan,
            sa_comp=np.nan,
            sq_comp=np.nan,
            sa_diff=np.nan,
            sq_diff=np.nan,
            ds_ref_norm=np.nan,
            ds_comp_norm=np.nan,
            ds_combined=np.nan,
        )

    # Position shift in meters
    position_shift = best_shift * pixel_size

    # Overlap length in meters
    overlap_length = len(best_ref_overlap) * pixel_size

    # Overlap ratio (relative to shorter profile)
    # Note: Use original lengths before scaling for overlap ratio calculation
    shorter_profile_length = min(len_ref, len_comp_original) * pixel_size
    overlap_ratio = overlap_length / shorter_profile_length

    # Compute roughness parameters (in meters)
    # Sa = mean absolute height
    # Sq = RMS roughness
    sa_ref = float(np.mean(np.abs(best_ref_overlap)))
    sq_ref = float(np.sqrt(np.mean(best_ref_overlap ** 2)))

    sa_comp = float(np.mean(np.abs(best_comp_overlap)))
    sq_comp = float(np.sqrt(np.mean(best_comp_overlap ** 2)))

    # Compute difference profile
    p_diff = best_comp_overlap - best_ref_overlap
    sa_diff = float(np.mean(np.abs(p_diff)))
    sq_diff = float(np.sqrt(np.mean(p_diff ** 2)))

    # Compute signature differences (dimensionless ratios)
    with np.errstate(divide="ignore", invalid="ignore"):
        ds_ref_norm = (sq_diff / sq_ref) ** 2 if sq_ref != 0 else np.nan
        ds_comp_norm = (sq_diff / sq_comp) ** 2 if sq_comp != 0 else np.nan
        ds_combined = (
            sq_diff ** 2 / (sq_ref * sq_comp)
            if (sq_ref != 0 and sq_comp != 0)
            else np.nan
        )

    # Create final results
    return ComparisonResults(
        is_profile_comparison=True,
        pixel_size_ref=pixel_size,
        pixel_size_comp=pixel_size,  # Same after equalization
        position_shift=position_shift,
        scale_factor=1.0 / best_scale,  # Return INVERSE of applied scale
        similarity_value=best_correlation,
        overlap_length=overlap_length,
        overlap_ratio=overlap_ratio,
        correlation_coefficient=best_correlation,
        sa_ref=sa_ref,
        sq_ref=sq_ref,
        sa_comp=sa_comp,
        sq_comp=sq_comp,
        sa_diff=sa_diff,
        sq_diff=sq_diff,
        ds_ref_norm=ds_ref_norm,
        ds_comp_norm=ds_comp_norm,
        ds_combined=ds_combined,
    )