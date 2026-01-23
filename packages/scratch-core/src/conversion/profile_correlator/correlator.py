"""Main entry point for profile correlation.

This module provides the primary interface for comparing striated mark profiles.
It handles the complete workflow including sampling equalization, length matching,
and multi-scale alignment.

All length and height measurements are in meters (SI units).
"""

import numpy as np

from conversion.profile_correlator.alignment import (
    align_partial_profile_multiscale,
    align_profiles_multiscale,
)
from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    ComparisonResults,
    Profile,
)
from conversion.profile_correlator.similarity import compute_comparison_metrics
from conversion.profile_correlator.transforms import (
    equalize_pixel_scale,
    make_profiles_equal_length,
)


def correlate_profiles(
    profile_ref: Profile,
    profile_comp: Profile,
    params: AlignmentParameters = AlignmentParameters(),
) -> ComparisonResults:
    """
    Compare two striated mark profiles and compute similarity metrics.

    It performs the complete comparison workflow:

    1. **Equalize sampling distances**: If the two profiles have different pixel
       sizes, the higher-resolution one is resampled to match the lower resolution.

    2. **Check for partial matching**: If the profiles differ in length by more
       than the partial_mark_threshold percentage, partial profile matching is
       used (brute-force candidate search followed by alignment).

    3. **Full or partial alignment**: Based on the length difference, either:
       - Full match: Profiles are cropped to equal length, then aligned
       - Partial match: The shorter profile is searched within the longer one

    4. **Compute metrics**: After alignment, comprehensive comparison metrics
       are calculated including correlation coefficient, roughness parameters,
       and signature differences.

    All measurements are in meters (SI units).

    :param profile_ref: Reference profile to compare against.
    :param profile_comp: Compared profile to align to the reference.
    :param params: Alignment parameters. If None, default parameters are used.
    :returns: ComparisonResults containing all computed metrics including:
        - correlation_coefficient: Pearson correlation after alignment
        - position_shift: Translation applied (m)
        - scale_factor: Scaling factor applied
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

    For best results, ensure profiles are preprocessed with appropriate
    filtering before calling this function.
    """
    # Step 1: Equalize pixel scale
    profile_ref_equal, profile_comp_equal = equalize_pixel_scale(
        profile_ref, profile_comp
    )

    # Get pixel size in meters
    pixel_size = profile_ref_equal.pixel_size

    # Step 2: Determine profile lengths and length difference
    size_1 = profile_ref_equal.length
    size_2 = profile_comp_equal.length

    # Compute relative length difference percentage
    length_diff_percent = abs(size_1 - size_2) / max(size_1, size_2) * 100

    # Step 3: Determine if full or partial matching is needed
    is_partial_profile = length_diff_percent >= params.partial_mark_threshold

    if not is_partial_profile:
        # Full profile comparison
        # Make profiles equal length by symmetric cropping
        profile_ref_mod, profile_comp_mod = make_profiles_equal_length(
            profile_ref_equal, profile_comp_equal
        )

        # Run multi-scale alignment
        alignment_result = align_profiles_multiscale(
            profile_ref_mod, profile_comp_mod, params
        )

        # Extract aligned profiles
        profile_ref_aligned = alignment_result.reference_aligned
        profile_comp_aligned = alignment_result.compared_aligned
        transforms = alignment_result.transforms

        partial_start_position = np.nan

    else:
        # Partial profile comparison
        # Determine which profile is longer (reference should be longer)
        if size_1 > size_2:
            # Reference is longer - normal case
            alignment_result, start_position = align_partial_profile_multiscale(
                profile_ref_equal, profile_comp_equal, params
            )
            profile_ref_aligned = alignment_result.reference_aligned
            profile_comp_aligned = alignment_result.compared_aligned
            transforms = alignment_result.transforms
            partial_start_position = start_position * pixel_size
        else:
            # Compared profile is longer - swap for alignment, then swap back
            alignment_result, start_position = align_partial_profile_multiscale(
                profile_comp_equal, profile_ref_equal, params
            )
            # Note: The result has reference and compared swapped
            profile_ref_aligned = alignment_result.compared_aligned
            profile_comp_aligned = alignment_result.reference_aligned
            transforms = alignment_result.transforms
            partial_start_position = start_position * pixel_size

    # Step 4: Compute comparison metrics
    results = compute_comparison_metrics(
        transforms=transforms,
        profile_ref=profile_ref_aligned,
        profile_comp=profile_comp_aligned,
        pixel_size=pixel_size,
    )

    # Compute overlap ratio
    if size_1 >= size_2:
        shorter_length = size_2 * pixel_size
    else:
        shorter_length = size_1 * pixel_size

    overlap_ratio = (
        results.overlap_length / shorter_length if shorter_length > 0 else np.nan
    )

    # Create final results with additional fields
    return ComparisonResults(
        is_profile_comparison=True,
        is_partial_profile=is_partial_profile,
        pixel_size_ref=pixel_size,
        pixel_size_comp=pixel_size,  # Same after equalization
        position_shift=results.position_shift,
        scale_factor=results.scale_factor,
        partial_start_position=partial_start_position,
        similarity_value=results.similarity_value,
        overlap_length=results.overlap_length,
        overlap_ratio=overlap_ratio,
        correlation_coefficient=results.correlation_coefficient,
        sa_ref=results.sa_ref,
        sq_ref=results.sq_ref,
        sa_comp=results.sa_comp,
        sq_comp=results.sq_comp,
        sa_diff=results.sa_diff,
        sq_diff=results.sq_diff,
        ds_ref_norm=results.ds_ref_norm,
        ds_comp_norm=results.ds_comp_norm,
        ds_combined=results.ds_combined,
    )
