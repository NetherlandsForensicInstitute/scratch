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
    AlignmentInputs,
    AlignmentParameters,
    AlignmentResult,
    StriationComparisonResults,
    Profile,
    RoughnessMetrics,
)
from conversion.profile_correlator.transforms import equalize_pixel_scale
from conversion.profile_correlator.statistics import (
    compute_cross_correlation,
    compute_overlap_ratio,
    compute_roughness_sa,
    compute_roughness_sq,
    compute_normalized_square_based_roughness_differences,
)
from conversion.resample import resample_array_1d


def correlate_profiles(
    profile_reference: Profile,
    profile_compared: Profile,
    params: AlignmentParameters = AlignmentParameters(),
) -> StriationComparisonResults | None:
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

    :param profile_reference: Reference profile to compare against.
    :param profile_compared: Compared profile to align to the reference.
    :param params: Alignment parameters. Key parameters:
        - max_scaling: Maximum scaling deviation (e.g., 0.05 for ±5%)
        - min_overlap_distance: Minimum overlap distance in meters (default 200 μm)
    :returns: StriationComparisonResults containing all computed metrics, or None if no
        valid alignment could be found.
    """
    # Step 1: Prepare alignment inputs
    inputs = _prepare_alignment_inputs(profile_reference, profile_compared, params)
    if inputs is None:
        return None

    # Step 2: Find best alignment
    alignment = _find_best_alignment(
        inputs.heights_ref,
        inputs.heights_comp,
        inputs.scale_factors,
        inputs.min_overlap_samples,
    )
    if alignment is None:
        return None

    # Step 3: Compute and return metrics
    return _compute_metrics(
        alignment, inputs.pixel_size, len(inputs.heights_ref), len(inputs.heights_comp)
    )


def _prepare_alignment_inputs(
    profile_reference: Profile,
    profile_compared: Profile,
    params: AlignmentParameters,
) -> AlignmentInputs | None:
    """
    Equalize pixel scales, compute minimum overlap, and generate scale factors.

    Returns None if either profile is too short for the minimum overlap requirement.

    :param profile_reference: Reference profile.
    :param profile_compared: Compared profile.
    :param params: Alignment parameters.
    :returns: Prepared alignment inputs, or None if profiles are too short.
    """
    profile_reference_eq, profile_compared_eq = equalize_pixel_scale(
        profile_reference, profile_compared
    )
    pixel_size = profile_reference_eq.pixel_size

    # Minimum overlap in samples
    min_overlap_samples = int(np.ceil(params.min_overlap_distance / pixel_size))

    # Early exit if either profile is too short
    if (
        len(profile_reference_eq.heights) < min_overlap_samples
        or len(profile_compared_eq.heights) < min_overlap_samples
    ):
        return None

    # Generate scale factors to try (7 steps gives ~1.7% intervals for ±5%)
    scale_factors = np.linspace(
        1.0 - params.max_scaling, 1.0 + params.max_scaling, params.n_scale_steps
    )

    # Make scaling symmetric to what you choose as reference or compared
    scale_factors = np.unique(np.concatenate((scale_factors, 1 / scale_factors)))

    return AlignmentInputs(
        heights_ref=profile_reference_eq.heights,
        heights_comp=profile_compared_eq.heights,
        pixel_size=pixel_size,
        scale_factors=scale_factors,
        min_overlap_samples=min_overlap_samples,
    )


def _calculate_idx_parameters(
    shift: int, len_small: int, len_large: int
) -> tuple[int, int, int]:
    """
    Find starting index for both striations, and compute overlap length
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
    heights_reference: FloatArray1D,
    heights_compared: FloatArray1D,
    scale_factors: FloatArray1D,
    min_overlap_samples: int,
) -> AlignmentResult | None:
    """
    Find the best alignment between two profiles using brute-force search.

    Tries all combinations of scale factors and shifts, returning the one
    with maximum correlation.

    :param heights_reference: Reference profile heights.
    :param heights_compared: Comparison profile heights.
    :param scale_factors: Array of scale factors to try.
    :param min_overlap_samples: Minimum required overlap in samples.
    :returns: Best alignment result, or None if no valid alignment found.
    """
    len_reference = len(heights_reference)

    best_correlation = -np.inf
    best_shift = None
    best_scale = None

    for scale in scale_factors:
        heights_compared_scaled = resample_array_1d(heights_compared, scale)
        len_compared = len(heights_compared_scaled)
        # Calculate shift range (ensure minimum overlap)
        min_shift = -(len_compared - min_overlap_samples)
        max_shift = len_reference - min_overlap_samples

        for shift in range(min_shift, max_shift + 1):
            idx_compared_start, idx_reference_start, overlap_length = (
                _calculate_idx_parameters(shift, len_compared, len_reference)
            )  # Calculate overlap region for this shift

            if overlap_length < min_overlap_samples:
                continue

            partial_reference = heights_reference[
                idx_reference_start : idx_reference_start + overlap_length
            ]
            partial_compared = heights_compared_scaled[
                idx_compared_start : idx_compared_start + overlap_length
            ]

            correlation = compute_cross_correlation(partial_reference, partial_compared)

            if correlation and correlation > best_correlation:
                best_correlation = correlation
                best_shift = shift
                best_scale = scale

    if best_shift is None or best_scale is None:
        return None

    # Redo computations for best_scale and best_shift (instead of copying partial_reference and partial_compared above multiple times. This saves time.)
    heights_compared_scaled = resample_array_1d(heights_compared, best_scale)
    idx_compared_start, idx_reference_start, overlap_length = _calculate_idx_parameters(
        best_shift, len(heights_compared_scaled), len_reference
    )

    best_reference_overlap = heights_reference[
        idx_reference_start : idx_reference_start + overlap_length
    ]
    best_compared_overlap = heights_compared_scaled[
        idx_compared_start : idx_compared_start + overlap_length
    ]

    return AlignmentResult(
        correlation=best_correlation,
        shift=best_shift,
        scale=best_scale,
        ref_overlap=best_reference_overlap,
        comp_overlap=best_compared_overlap,
    )


def _compute_metrics(
    alignment: AlignmentResult,
    pixel_size: float,
    len_reference: int,
    len_compared: int,
) -> StriationComparisonResults:
    """
    Compute comparison metrics from an alignment result.

    :param alignment: The best alignment found.
    :param pixel_size: Pixel size in meters.
    :param len_reference: Length of reference profile in samples.
    :param len_compared: Length of comparison profile in samples.
    :returns: Full comparison results.
    """
    reference_overlap = alignment.ref_overlap
    compared_overlap = alignment.comp_overlap

    # Convert to meters
    position_shift = alignment.shift * pixel_size
    overlap_length = len(reference_overlap) * pixel_size
    reference_length = len_reference * pixel_size
    compared_length = len_compared * pixel_size

    overlap_ratio = compute_overlap_ratio(
        overlap_length, reference_length, compared_length
    )

    # Roughness metrics
    sa_reference = compute_roughness_sa(reference_overlap)
    mean_square_reference = compute_roughness_sq(reference_overlap)
    sa_compared = compute_roughness_sa(compared_overlap)
    mean_square_compared = compute_roughness_sq(compared_overlap)

    # Difference profile roughness
    diff_profile = compared_overlap - reference_overlap
    sa_diff = compute_roughness_sa(diff_profile)
    mean_square_of_difference = compute_roughness_sq(diff_profile)

    # Signature differences
    roughness = RoughnessMetrics(
        mean_square_ref=mean_square_reference,
        mean_square_comp=mean_square_compared,
        mean_square_of_difference=mean_square_of_difference,
    )
    signature_diff = compute_normalized_square_based_roughness_differences(roughness)

    return StriationComparisonResults(
        pixel_size=pixel_size,
        position_shift=position_shift,
        scale_factor=alignment.scale,
        similarity_value=alignment.correlation,
        overlap_length=overlap_length,
        overlap_ratio=overlap_ratio,
        correlation_coefficient=alignment.correlation,
        sa_ref=sa_reference,
        mean_square_ref=mean_square_reference,
        sa_comp=sa_compared,
        mean_square_comp=mean_square_compared,
        sa_diff=sa_diff,
        mean_square_of_difference=mean_square_of_difference,
        ds_roughness_normalized_to_reference=signature_diff.roughness_normalized_to_reference,
        ds_roughness_normalized_to_compared=signature_diff.roughness_normalized_to_compared,
        ds_roughness_normalized_to_reference_and_compared=signature_diff.roughness_normalized_to_reference_and_compared,
    )
