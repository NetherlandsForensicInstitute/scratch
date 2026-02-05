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
    AlignmentResult,
    ComparisonResults,
    Profile,
    RoughnessMetrics,
)
from conversion.profile_correlator.transforms import equalize_pixel_scale
from conversion.profile_correlator.statistics import (
    compute_cross_correlation,
    compute_overlap_ratio,
    compute_roughness_sa,
    compute_roughness_sq,
    compute_signature_differences,
)
from conversion.resample import resample_array_1d


def _calculate_idx_parameters(
    shift: int, len_small: int, len_large: int
) -> tuple[int, int, int]:
    """
    Find starting idx for both striations, and compute overlap length
    """

    if shift >= 0:
        idx_large_start = shift
        idx_small_start = 0
        overlap_length = min(len_large - shift, len_small)
    else:
        idx_large_start = 0
        idx_small_start = -shift
        overlap_length = min(len_large, len_small + shift)

    return idx_small_start, idx_large_start, overlap_length


def _find_best_alignment(
    heights_ref: FloatArray1D,
    heights_comp: FloatArray1D,
    scale_factors: FloatArray1D,
    min_overlap_samples: int,
) -> AlignmentResult | None:
    """
    Find the best alignment between two profiles using brute-force search.

    Tries all combinations of scale factors and shifts, returning the one
    with maximum correlation.

    :param heights_ref: Reference profile heights.
    :param heights_comp: Comparison profile heights.
    :param scale_factors: Array of scale factors to try.
    :param min_overlap_samples: Minimum required overlap in samples.
    :returns: Best alignment result, or None if no valid alignment found.
    """
    len_ref = len(heights_ref)

    best_correlation = -np.inf
    best_shift = None
    best_scale = None
    best_ref_overlap = None
    best_comp_overlap = None

    for scale in scale_factors:
        heights_comp_scaled = resample_array_1d(heights_comp, scale)
        len_comp = len(heights_comp_scaled)
        # Calculate shift range (ensure minimum overlap)
        min_shift = -(len_comp - min_overlap_samples)
        max_shift = len_ref - min_overlap_samples

        for shift in range(min_shift, max_shift + 1):
            idx_comp_start, idx_ref_start, overlap_length = _calculate_idx_parameters(
                shift, len_comp, len_ref
            )  # Calculate overlap region for this shift

            if overlap_length < min_overlap_samples:
                continue

            partial_ref = heights_ref[idx_ref_start : idx_ref_start + overlap_length]
            partial_comp = heights_comp_scaled[
                idx_comp_start : idx_comp_start + overlap_length
            ]

            correlation = compute_cross_correlation(partial_ref, partial_comp)

            if not np.isnan(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_shift = shift
                best_scale = scale
                best_ref_overlap = partial_ref.copy()
                best_comp_overlap = partial_comp.copy()

    if best_ref_overlap is None or best_comp_overlap is None:
        return None

    return AlignmentResult(
        correlation=best_correlation,
        shift=best_shift,
        scale=best_scale,
        ref_overlap=best_ref_overlap,
        comp_overlap=best_comp_overlap,
    )


def _compute_metrics(
    alignment: AlignmentResult,
    pixel_size: float,
    len_ref: int,
    len_comp: int,
) -> ComparisonResults:
    """
    Compute comparison metrics from an alignment result.

    :param alignment: The best alignment found.
    :param pixel_size: Pixel size in meters.
    :param len_ref: Length of reference profile in samples.
    :param len_comp: Length of comparison profile in samples.
    :returns: Full comparison results.
    """
    ref_overlap = alignment.ref_overlap
    comp_overlap = alignment.comp_overlap

    # Convert to meters
    position_shift = alignment.shift * pixel_size
    overlap_length = len(ref_overlap) * pixel_size
    ref_length = len_ref * pixel_size
    comp_length = len_comp * pixel_size

    overlap_ratio = compute_overlap_ratio(overlap_length, ref_length, comp_length)

    # Roughness metrics
    sa_ref = compute_roughness_sa(ref_overlap)
    sq_ref = compute_roughness_sq(ref_overlap)
    sa_comp = compute_roughness_sa(comp_overlap)
    sq_comp = compute_roughness_sq(comp_overlap)

    # Difference profile roughness
    diff_profile = comp_overlap - ref_overlap
    sa_diff = compute_roughness_sa(diff_profile)
    sq_diff = compute_roughness_sq(diff_profile)

    # Signature differences
    roughness = RoughnessMetrics(sq_ref=sq_ref, sq_comp=sq_comp, sq_diff=sq_diff)
    signature_diff = compute_signature_differences(roughness)

    return ComparisonResults(
        is_profile_comparison=True,
        pixel_size_ref=pixel_size,
        pixel_size_comp=pixel_size,
        position_shift=position_shift,
        scale_factor=1.0 / alignment.scale,
        similarity_value=alignment.correlation,
        overlap_length=overlap_length,
        overlap_ratio=overlap_ratio,
        correlation_coefficient=alignment.correlation,
        sa_ref=sa_ref,
        sq_ref=sq_ref,
        sa_comp=sa_comp,
        sq_comp=sq_comp,
        sa_diff=sa_diff,
        sq_diff=sq_diff,
        ds_ref_norm=signature_diff.ref_norm,
        ds_comp_norm=signature_diff.comp_norm,
        ds_combined=signature_diff.combined,
    )


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

    heights_ref = profile_ref_eq.heights
    heights_comp = profile_comp_eq.heights
    len_ref = len(heights_ref)
    len_comp = len(heights_comp)

    # Minimum overlap in samples
    min_overlap_samples = int(np.ceil(params.min_overlap_distance / pixel_size))

    # Early exit if either profile is too short
    if len_ref < min_overlap_samples or len_comp < min_overlap_samples:
        return None

    # Generate scale factors to try (7 steps gives ~1.7% intervals for ±5%)
    scale_factors = np.linspace(1.0 - params.max_scaling, 1.0 + params.max_scaling, 7)

    # make scaling symmetric to what you choose as ref or comp
    scale_factors = np.unique(np.concatenate((scale_factors, 1 / scale_factors)))

    # Step 2: Find best alignment
    alignment = _find_best_alignment(
        heights_ref, heights_comp, scale_factors, min_overlap_samples
    )
    if alignment is None:
        return None

    # Step 3: Compute and return metrics
    return _compute_metrics(alignment, pixel_size, len_ref, len_comp)
