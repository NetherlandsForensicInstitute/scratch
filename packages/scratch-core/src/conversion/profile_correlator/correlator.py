"""Main entry point for profile correlation.

This module provides the primary interface for comparing striated mark profiles.
It handles the complete workflow including sampling equalization, length matching,
and multi-scale alignment.

The main function is:
- correlate_profiles: Compare two profiles and compute similarity metrics

This corresponds to the MATLAB function:
- ProfileCorrelatorSingle.m
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
    equalize_sampling_distance,
    make_profiles_equal_length,
)


def correlate_profiles(
    profile_ref: Profile,
    profile_comp: Profile,
    params: AlignmentParameters | None = None,
) -> ComparisonResults:
    """
    Compare two striated mark profiles and compute similarity metrics.

    This function is the main entry point for profile correlation. It performs
    the complete comparison workflow:

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

    This corresponds to MATLAB's ProfileCorrelatorSingle.m.

    :param profile_ref: Reference profile to compare against.
    :param profile_comp: Compared profile to align to the reference.
    :param params: Alignment parameters. If None, default parameters are used.
    :returns: ComparisonResults containing all computed metrics including:
        - correlation_coefficient: Pearson correlation after alignment
        - position_shift: Translation applied (in micrometers)
        - scale_factor: Scaling factor applied
        - sa_ref, sq_ref: Roughness metrics for reference
        - sa_comp, sq_comp: Roughness metrics for compared profile
        - ds_combined: Combined signature difference

    Example::

        >>> import numpy as np
        >>> from conversion.profile_correlator.data_types import Profile
        >>> # Create two similar profiles
        >>> x = np.linspace(0, 10, 1000)
        >>> ref = Profile(np.sin(x) + 0.01 * np.random.randn(1000), pixel_size=0.5e-6)
        >>> comp = Profile(np.sin(x - 0.05) + 0.01 * np.random.randn(1000), pixel_size=0.5e-6)
        >>> results = correlate_profiles(ref, comp)
        >>> results.correlation_coefficient > 0.9
        True

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
    # Use default parameters if not provided
    if params is None:
        params = AlignmentParameters()

    # Step 1: Equalize sampling distances
    # MATLAB: [profile_ref_equal, profile_comp_equal] = EqualizeSamplingDistance(...)
    profile_ref_equal, profile_comp_equal = equalize_sampling_distance(
        profile_ref, profile_comp
    )

    # Get pixel size in micrometers
    pixel_size_um = profile_ref_equal.pixel_size * 1e6

    # Step 2: Determine profile lengths and length difference
    size_1 = profile_ref_equal.length
    size_2 = profile_comp_equal.length

    # Compute relative length difference percentage
    # MATLAB: lengthDiffPercentage = abs(size_1 - size_2) / max(size_1, size_2) * 100
    length_diff_percent = abs(size_1 - size_2) / max(size_1, size_2) * 100

    # Step 3: Determine if full or partial matching is needed
    is_partial_profile = length_diff_percent >= params.partial_mark_threshold

    if not is_partial_profile:
        # Full profile comparison
        # Make profiles equal length by symmetric cropping
        # MATLAB: [profile_ref_equal_mod, profile_comp_equal_mod] = MakeDatasetLengthEqual(...)
        profile_ref_mod, profile_comp_mod = make_profiles_equal_length(
            profile_ref_equal, profile_comp_equal
        )

        # Since we passed Profile objects, we get Profile objects back
        assert isinstance(profile_ref_mod, Profile)
        assert isinstance(profile_comp_mod, Profile)

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
            partial_start_position = start_position * pixel_size_um
        else:
            # Compared profile is longer - swap for alignment, then swap back
            alignment_result, start_position = align_partial_profile_multiscale(
                profile_comp_equal, profile_ref_equal, params
            )
            # Note: The result has reference and compared swapped
            profile_ref_aligned = alignment_result.compared_aligned
            profile_comp_aligned = alignment_result.reference_aligned
            transforms = alignment_result.transforms
            partial_start_position = start_position * pixel_size_um

    # Step 4: Compute comparison metrics
    # MATLAB: results_table = GetStriatedMarkComparisonResults(...)
    results = compute_comparison_metrics(
        transforms=transforms,
        profile_ref=profile_ref_aligned,
        profile_comp=profile_comp_aligned,
        pixel_size_um=pixel_size_um,
    )

    # Compute overlap ratio
    # MATLAB: if size_1 >= size_2:
    #             results_table.pOverlap = lOverlap/(size_2*vPixSep2)
    #         else:
    #             results_table.pOverlap = lOverlap/(size_1*vPixSep1)
    if size_1 >= size_2:
        shorter_length_um = size_2 * pixel_size_um
    else:
        shorter_length_um = size_1 * pixel_size_um

    overlap_ratio = (
        results.overlap_length / shorter_length_um if shorter_length_um > 0 else np.nan
    )

    # Create final results with additional fields
    return ComparisonResults(
        is_profile_comparison=True,
        is_partial_profile=is_partial_profile,
        pixel_size_ref=pixel_size_um,
        pixel_size_comp=pixel_size_um,  # Same after equalization
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
