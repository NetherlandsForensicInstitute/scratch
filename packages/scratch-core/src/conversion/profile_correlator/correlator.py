"""
Main entry point for profile correlation.

This module provides the primary interface for comparing striated mark profiles.
It uses a brute-force approach: try all possible shifts and scales between
profiles and select the one with maximum cross-correlation, subject to a
minimum overlap constraint.

All length and height measurements are in meters (SI units).

Algorithm Overview
------------------
The algorithm uses a global brute-force search strategy:

1. **Equalizes pixel scales** between profiles (downsample to common resolution)
2. **Tries multiple scale factors** (e.g., 0.95, 0.97, ..., 1.03, 1.05)
3. **For each scale, tries all shifts** with at least min_overlap_distance overlap
4. **Computes correlation** at each shift position
5. **Selects maximum** correlation as the optimal alignment

This approach is guaranteed to find the global maximum correlation, which may
be at a position far from zero shift for repetitive patterns.
"""

import numpy as np

from container_models.base import FloatArray1D
from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    ComparisonResults,
    Profile,
)
from conversion.profile_correlator.transforms import apply_scaling, equalize_pixel_scale
from conversion.profile_correlator.statistics import (
    compute_overlap_ratio,
    compute_roughness_sa,
    compute_roughness_sq,
    compute_signature_differences,
)


def _compute_correlation(
    profile_ref: FloatArray1D,
    profile_comp: FloatArray1D,
) -> float:
    """
    Compute Pearson correlation between two profile segments.

    :param profile_ref: Reference profile segment (1D array).
    :param profile_comp: Comparison profile segment (1D array).
    :returns: Pearson correlation coefficient, or NaN if computation fails.
    """
    valid_mask = ~(np.isnan(profile_ref) | np.isnan(profile_comp))

    if np.sum(valid_mask) < 2:
        return np.nan

    return float(np.corrcoef(profile_ref[valid_mask], profile_comp[valid_mask])[0, 1])


def correlate_profiles(
    profile_ref: Profile,
    profile_comp: Profile,
    params: AlignmentParameters = AlignmentParameters(),
) -> ComparisonResults | None:
    """
    Compare two striated mark profiles and compute similarity metrics.

    This function performs profile alignment using a brute-force approach:
    it tries multiple scale factors and all possible shifts between the profiles
    (subject to a minimum overlap constraint) and selects the combination with
    maximum cross-correlation.

    The workflow is:

    1. **Equalize sampling distances**: If the two profiles have different pixel
       sizes, the higher-resolution one is downsampled to match the lower resolution.

    2. **Brute-force search**: Try all shift/scale combinations and compute
       cross-correlation at each position. For each scale factor in the range
       defined by params.max_scaling, determine valid shifts that maintain at
       least min_overlap_distance of overlap, and select the shift/scale with
       maximum correlation.

    3. **Compute metrics**: Calculate comprehensive comparison metrics including
       correlation coefficient, roughness parameters, and signature differences.

    All measurements are in meters (SI units).

    :param profile_ref: Reference profile to compare against.
    :param profile_comp: Compared profile to align to the reference.
    :param params: Alignment parameters. Key parameters:
        - max_scaling: Maximum scaling deviation (e.g., 0.05 for ±5%)
        - min_overlap_distance: Minimum overlap distance in meters (default 200 μm)
    :returns: ComparisonResults containing all computed metrics, or None if no
        valid alignment could be found.
    """
    # Step 1: Equalize pixel scales
    profile_ref_eq, profile_comp_eq = equalize_pixel_scale(profile_ref, profile_comp)
    pixel_size = profile_ref_eq.pixel_size

    # Get 1D profile data
    ref_data = profile_ref_eq.heights
    comp_data = profile_comp_eq.heights

    len_ref = len(ref_data)
    len_comp = len(comp_data)

    # Minimum overlap of the two Profiles in pixels
    # Use ceil to ensure we meet the minimum distance (int truncates, which could give less)
    min_overlap_samples = int(np.ceil(params.min_overlap_distance / pixel_size))

    # Generate scale factors to try.
    # The MATLAB implementation uses fminsearchbnd to jointly optimize shift and scale
    # as a 2D continuous optimization problem. This Python version simplifies to a
    # brute-force grid search: iterate over discrete scale factors, and for each scale
    # try all possible shifts. 7 scale steps gives ~1.7% intervals for the default ±5%
    # range, balancing accuracy vs computation time.
    num_scale_steps = 7
    scale_factors = np.linspace(
        1.0 - params.max_scaling, 1.0 + params.max_scaling, num_scale_steps
    )

    # Step 2: Global brute-force search over all scales and shifts
    best_correlation = -np.inf
    best_shift = 0
    best_scale = 1.0
    best_ref_overlap: FloatArray1D | None = None
    best_comp_overlap: FloatArray1D | None = None

    for scale in scale_factors:
        # Apply scaling to comparison profile
        comp_data_scaled = apply_scaling(comp_data, scale)
        len_comp = len(comp_data_scaled)

        # Skip if profiles too short for minimum overlap
        if len_comp < min_overlap_samples or len_ref < min_overlap_samples:
            continue

        # Determine which profile is larger
        if len_ref >= len_comp:
            large = ref_data
            small = comp_data_scaled
            ref_is_large = True
        else:
            large = comp_data_scaled
            small = ref_data
            ref_is_large = False

        len_large = len(large)
        len_small = len(small)

        # Calculate shift range (ensure minimum overlap)
        min_shift = -(len_small - min_overlap_samples)
        max_shift = len_large - min_overlap_samples

        # Try all shifts in range
        for shift in range(min_shift, max_shift + 1):
            # Calculate overlap region for this shift
            # Positive shift: large profile shifts left (or equivalently, small shifts right)
            if shift >= 0:
                idx_large_start = shift
                idx_small_start = 0
                overlap_len = min(len_large - shift, len_small)
            else:
                idx_large_start = 0
                idx_small_start = -shift
                overlap_len = min(len_large, len_small + shift)

            if overlap_len < min_overlap_samples:
                continue

            # Extract overlapping segments
            partial_large = large[idx_large_start : idx_large_start + overlap_len]
            partial_small = small[idx_small_start : idx_small_start + overlap_len]

            if ref_is_large:
                partial_ref = partial_large
                partial_comp = partial_small
            else:
                partial_ref = partial_small
                partial_comp = partial_large

            correlation = _compute_correlation(partial_ref, partial_comp)

            if not np.isnan(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_shift = shift
                best_scale = scale
                best_ref_overlap = partial_ref.copy()
                best_comp_overlap = partial_comp.copy()

    # Step 3: Compute metrics for the best alignment
    if best_ref_overlap is None or best_comp_overlap is None:
        return None

    # Position shift in meters
    position_shift = best_shift * pixel_size

    # Overlap length in meters
    overlap_length = len(best_ref_overlap) * pixel_size

    # Overlap ratio (relative to shorter profile)
    ref_length = len_ref * pixel_size
    comp_length = len_comp * pixel_size
    overlap_ratio = compute_overlap_ratio(overlap_length, ref_length, comp_length)

    # Compute roughness parameters (in meters)
    sa_ref = compute_roughness_sa(best_ref_overlap)
    sq_ref = compute_roughness_sq(best_ref_overlap)

    sa_comp = compute_roughness_sa(best_comp_overlap)
    sq_comp = compute_roughness_sq(best_comp_overlap)

    # Compute difference profile roughness
    p_diff = best_comp_overlap - best_ref_overlap
    sa_diff = compute_roughness_sa(p_diff)
    sq_diff = compute_roughness_sq(p_diff)

    # Compute signature differences (dimensionless ratios)
    ds_ref_norm, ds_comp_norm, ds_combined = compute_signature_differences(
        sq_diff, sq_ref, sq_comp
    )

    # Create final results
    return ComparisonResults(
        is_profile_comparison=True,
        pixel_size_ref=pixel_size,
        pixel_size_comp=pixel_size,
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
