"""Multi-scale profile alignment algorithms.

This module provides functions for aligning 1D profiles using multi-scale
coarse-to-fine registration. The algorithm iteratively refines the alignment
at progressively finer scales to achieve accurate registration while avoiding
local minima.

The main functions are:
- align_profiles_multiscale: Full profile multi-scale alignment
- align_partial_profile_multiscale: Partial profile alignment with candidate search

These correspond to the MATLAB functions:
- AlignInterProfilesMultiScale.m
- AlignInterProfilesPartialMultiScale.m
- ErrorfuncForInterProfileAlignment.m
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    AlignmentResult,
    Profile,
    TransformParameters,
)
from conversion.profile_correlator.filtering import apply_lowpass_filter_1d
from conversion.profile_correlator.similarity import compute_cross_correlation
from conversion.profile_correlator.transforms import (
    apply_transform,
    compute_cumulative_transform,
    remove_boundary_zeros,
)


def _alignment_objective(
    x: NDArray[np.floating],
    profile_ref: NDArray[np.floating],
    profile_comp: NDArray[np.floating],
) -> float:
    """
    Objective function for profile alignment optimization.

    This function computes the negative cross-correlation between the reference
    profile and a transformed version of the compared profile. The transformation
    consists of translation and scaling.

    The scaling parameter is encoded as (scale - 1) * 10000 to balance the
    influence of translation and scaling on the objective. This ensures that
    approximately 1 sample of translation has a similar effect on the correlation
    as 0.0001 in scaling.

    This corresponds to MATLAB's ErrorfuncForInterProfileAlignment.m.

    :param x: Optimization variables [translation, encoded_scaling] where
        encoded_scaling = (scaling - 1) * 10000.
    :param profile_ref: Reference profile (1D array).
    :param profile_comp: Compared profile to be transformed (1D array).
    :returns: Negative cross-correlation (minimized during optimization).
    """
    # Extract and decode parameters
    # MATLAB: translation = x(1); scaling = x(2)/10000 + 1
    translation = x[0]
    scaling = x[1] / 10000.0 + 1.0

    # Create transform and apply to compared profile
    transform = TransformParameters(translation=translation, scaling=scaling)
    profile_comp_transformed = apply_transform(profile_comp, transform)

    # Compute similarity (cross-correlation)
    correlation = compute_cross_correlation(profile_ref, profile_comp_transformed)

    # Return negative because we minimize (want to maximize correlation)
    # Handle NaN case - return large positive value (bad correlation)
    if np.isnan(correlation):
        return 1.0
    return -correlation


def align_profiles_multiscale(
    profile_ref: Profile,
    profile_comp: Profile,
    params: AlignmentParameters | None = None,
) -> AlignmentResult:
    """
    Align two profiles using multi-scale coarse-to-fine registration.

    This function performs iterative alignment starting from coarse scales
    (large cutoff wavelengths) and progressively refining at finer scales.
    At each scale level:
    1. Both profiles are low-pass filtered at the current cutoff wavelength
    2. Subsampling is applied for computational efficiency at coarse scales
    3. Translation and scaling are optimized to maximize cross-correlation
    4. The compared profile is transformed by the optimized parameters
    5. The process continues to the next finer scale

    The algorithm terminates when all scale levels have been processed, then
    boundary zeros are optionally removed from both profiles.

    Scale levels that are below the resolution limit or outside the cutoff
    bounds are skipped.

    This corresponds to MATLAB's AlignInterProfilesMultiScale.m.

    :param profile_ref: Reference profile (kept fixed during alignment).
    :param profile_comp: Compared profile (transformed to align with reference).
    :param params: Alignment parameters. If None, default parameters are used.
    :returns: AlignmentResult containing the sequence of transforms, correlation
        history, final correlation, and the aligned profiles.
    :raises ValueError: If profiles have different lengths. Use
        make_profiles_equal_length() first if needed.

    Example::

        >>> import numpy as np
        >>> from conversion.profile_correlator.data_types import Profile
        >>> # Create two profiles with a known shift
        >>> x = np.linspace(0, 10, 1000)
        >>> ref = Profile(np.sin(x), pixel_size=0.5e-6)
        >>> comp = Profile(np.sin(x - 0.1), pixel_size=0.5e-6)  # Shifted
        >>> result = align_profiles_multiscale(ref, comp)
        >>> result.final_correlation > 0.99
        True
    """
    # Use default parameters if not provided
    if params is None:
        params = AlignmentParameters()

    # Extract depth data and compute mean profiles if multi-column
    profile_1 = profile_ref.mean_profile(use_mean=params.use_mean)
    profile_2 = profile_comp.mean_profile(use_mean=params.use_mean)

    # Validate that profiles have the same length
    if len(profile_1) != len(profile_2):
        raise ValueError(
            f"Profiles must have the same length. "
            f"Got {len(profile_1)} and {len(profile_2)}. "
            "Use make_profiles_equal_length() first."
        )

    # Get pixel size and determine resolution limit
    # MATLAB: xdim = profiles1_in.xdim; LR = max(cutoff_lo * 1e-6, 2 * xdim)
    pixel_size = profile_ref.pixel_size  # In meters
    pixel_size_um = pixel_size * 1e6  # In micrometers

    if profile_ref.resolution_limit is not None:
        resolution_limit = profile_ref.resolution_limit
    else:
        resolution_limit = max(params.cutoff_lo * 1e-6, 2 * pixel_size)

    # Determine effective cutoff bounds
    # Use the more restrictive bounds from profiles and parameters
    cutoff_hi = params.cutoff_hi
    cutoff_lo = params.cutoff_lo

    if profile_ref.cutoff_hi is not None and profile_comp.cutoff_hi is not None:
        cutoff_hi = min(cutoff_hi, profile_ref.cutoff_hi, profile_comp.cutoff_hi)
    if profile_ref.cutoff_lo is not None and profile_comp.cutoff_lo is not None:
        cutoff_lo = max(cutoff_lo, profile_ref.cutoff_lo, profile_comp.cutoff_lo)

    # Convert max_translation from micrometers to samples
    # MATLAB: max_translation = max_translation/1000; then round(max_translation/xdim/1000)
    # which is max_translation_um / 1000 / (xdim_m * 1000) = max_translation_um / (xdim_m * 1e6)
    # = max_translation_um / pixel_size_um
    max_translation_samples = params.max_translation / pixel_size_um

    # Initialize tracking variables
    transforms: list[TransformParameters] = []
    correlation_history: list[tuple[float, float]] = []

    # Track cumulative transform applied to profile_2
    translation_total = 0.0
    scaling_total = 1.0
    current_scaling = 1.0  # For adjusting bounds at each scale

    # Working copy of compared profile
    profile_2_mod = profile_2.copy()

    # Process each scale level from coarse to fine
    for scale_idx, cutoff_um in enumerate(params.scale_passes):
        # Check if this scale level should be processed
        # MATLAB: pass(scale_level, 1) * 1e-6 >= resolution_threshold
        #         && pass(scale_level, 1) <= cutoff_hi
        #         && pass(scale_level, 1) >= cutoff_lo
        cutoff_m = cutoff_um * 1e-6

        if cutoff_m < resolution_limit:
            # Scale is finer than resolution limit, skip
            continue
        if cutoff_um > cutoff_hi:
            # Scale is coarser than high cutoff bound, skip
            continue
        if cutoff_um < cutoff_lo:
            # Scale is finer than low cutoff bound, skip
            continue

        # Apply low-pass filter to both profiles at current scale
        # MATLAB: result1 = ApplyLowPassFilter(mean_1, param_tmp)
        profile_1_filtered = apply_lowpass_filter_1d(
            profile_1,
            cutoff_wavelength_um=cutoff_um,
            pixel_size_m=pixel_size,
            cut_borders=params.cut_borders_after_smoothing,
        )

        profile_2_filtered = apply_lowpass_filter_1d(
            profile_2_mod,
            cutoff_wavelength_um=cutoff_um,
            pixel_size_m=pixel_size,
            cut_borders=params.cut_borders_after_smoothing,
        )

        # Compute subsampling factor for efficiency
        # MATLAB: subsample_factor = ceil(param_tmp.cutoff_lo / (xdim * 1e6) / 2 / 5)
        # Note: cutoff_lo here is the current scale's cutoff
        subsample_factor = max(1, int(np.ceil(cutoff_um / pixel_size_um / 2 / 5)))

        # Subsample for optimization (coarse scales)
        profile_1_subsampled = profile_1_filtered[::subsample_factor]
        profile_2_subsampled = profile_2_filtered[::subsample_factor]

        # Compute bounds for this scale level
        # MATLAB bounds logic:
        # max_translation_tmp = max_translation - translation_tot
        # min_translation_tmp = max_translation + translation_tot
        # max_scaling_tmp = max_scaling * (1 - (scaling_tot - 1))
        # min_scaling_tmp = max_scaling * (1 + (scaling_tot - 1))
        max_trans_adj = max_translation_samples - translation_total
        min_trans_adj = max_translation_samples + translation_total
        max_scaling_adj = params.max_scaling * (1 - (scaling_total - 1))
        min_scaling_adj = params.max_scaling * (1 + (scaling_total - 1))

        # Bounds in subsampled coordinates
        # MATLAB: lb = [-round(min_translation_tmp/subsample_factor)
        #               ((1 - min_scaling_tmp)/current_scaling - 1) * 10000]
        #         ub = [ round(max_translation_tmp/subsample_factor)
        #               ((1 + max_scaling_tmp)/current_scaling - 1) * 10000]
        trans_lb = -int(round(min_trans_adj / subsample_factor))
        trans_ub = int(round(max_trans_adj / subsample_factor))

        # Note: the scaling bounds account for current_scaling
        scale_lb = ((1 - min_scaling_adj) / current_scaling - 1) * 10000
        scale_ub = ((1 + max_scaling_adj) / current_scaling - 1) * 10000

        bounds = [(trans_lb, trans_ub), (scale_lb, scale_ub)]

        # Initial guess: start from previous iteration or zero
        x0 = np.array([0.0, 0.0])

        # Run bounded optimization
        # MATLAB uses fminsearchbnd; we use scipy.optimize.minimize with L-BFGS-B
        result = minimize(
            _alignment_objective,
            x0,
            args=(profile_1_subsampled, profile_2_subsampled),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-6,  # MATLAB: TolFun = 0.000001
                "gtol": 1e-6,  # MATLAB: TolX = 0.000001
                "maxiter": 1000,
                "disp": False,
            },
        )

        # Extract optimized parameters (in subsampled space)
        translation_subsampled = result.x[0]
        scaling_encoded = result.x[1]

        # Convert back to full resolution
        # MATLAB: translation = x(1) * subsample_factor
        #         scaling = x(2)/10000 + 1
        translation = translation_subsampled * subsample_factor
        scaling = scaling_encoded / 10000.0 + 1.0

        # Store transform for this scale level
        transform = TransformParameters(translation=translation, scaling=scaling)
        transforms.append(transform)

        # Update cumulative transform tracking
        current_scaling = scaling
        translation_total = translation_total + translation
        scaling_total = scaling_total * scaling

        # Apply transform to get aligned compared profile for correlation computation
        profile_2_transformed = apply_transform(profile_2_mod, transform)

        # Compute correlation at this scale level (on filtered profiles)
        profile_2_filtered_transformed = apply_transform(profile_2_filtered, transform)
        correlation_filtered = compute_cross_correlation(
            profile_1_filtered, profile_2_filtered_transformed
        )

        # Compute correlation on original (unfiltered) profiles after removing zeros
        profile_1_no_zeros, profile_2_no_zeros, _ = remove_boundary_zeros(
            profile_1, profile_2_transformed
        )
        correlation_original = compute_cross_correlation(
            profile_1_no_zeros, profile_2_no_zeros
        )

        correlation_history.append((correlation_filtered, correlation_original))

        # Update working copy of compared profile for next iteration
        profile_2_mod = profile_2_transformed

    # Handle case where no scales were processed
    if len(transforms) == 0:
        transforms = [TransformParameters(translation=0.0, scaling=1.0)]
        correlation_history = [(np.nan, np.nan)]

    # Apply all transforms to get final aligned profile
    profile_2_aligned = apply_transform(profile_2, transforms)

    # Optionally remove boundary zeros
    if params.remove_boundary_zeros:
        profile_1_aligned, profile_2_aligned, _ = remove_boundary_zeros(
            profile_1, profile_2_aligned
        )
    else:
        profile_1_aligned = profile_1

    # Compute final correlation on the output profiles
    final_correlation = compute_cross_correlation(profile_1_aligned, profile_2_aligned)

    # Update the last entry in correlation history with final value
    if len(correlation_history) > 0:
        correlation_history[-1] = (
            correlation_history[-1][0],
            final_correlation,
        )

    # Compute total transform
    total_translation, total_scaling = compute_cumulative_transform(transforms)

    # Build correlation history array
    correlation_array = np.array(correlation_history, dtype=np.float64)

    return AlignmentResult(
        transforms=tuple(transforms),
        correlation_history=correlation_array,
        final_correlation=final_correlation,
        reference_aligned=profile_1_aligned,
        compared_aligned=profile_2_aligned,
        total_translation=total_translation,
        total_scaling=total_scaling,
    )


def align_partial_profile_multiscale(
    reference: Profile,
    partial: Profile,
    params: AlignmentParameters | None = None,
    candidate_positions: Sequence[int] | None = None,
) -> tuple[AlignmentResult, int]:
    """
    Align a partial (shorter) profile to a reference using multi-scale registration.

    This function handles the case where the profiles have significantly different
    lengths (more than partial_mark_threshold percent difference). It performs:
    1. Candidate search to find potential alignment positions (if not provided)
    2. For each candidate, extract a reference segment and run full alignment
    3. Select the candidate with the highest final correlation

    If candidate_positions is provided, it uses those directly instead of
    running the candidate search algorithm.

    This corresponds to MATLAB's AlignInterProfilesPartialMultiScale.m.

    :param reference: Reference profile (longer one).
    :param partial: Partial profile (shorter one) to align to reference.
    :param params: Alignment parameters. If None, default parameters are used.
    :param candidate_positions: Optional list of starting positions to try.
        If None, positions are determined by candidate search.
    :returns: Tuple of (AlignmentResult, best_start_position) where
        best_start_position is the index in the reference where the partial
        profile best aligns.

    Example::

        >>> import numpy as np
        >>> from conversion.profile_correlator.data_types import Profile
        >>> # Create reference with partial segment
        >>> x = np.linspace(0, 20, 2000)
        >>> ref = Profile(np.sin(x), pixel_size=0.5e-6)
        >>> partial = Profile(np.sin(x[500:1000]), pixel_size=0.5e-6)
        >>> result, start = align_partial_profile_multiscale(ref, partial)
        >>> 450 < start < 550  # Should find position near 500
        True
    """
    # Use default parameters if not provided
    if params is None:
        params = AlignmentParameters()

    # Import here to avoid circular imports
    from conversion.profile_correlator.candidate_search import find_match_candidates

    # Get mean profiles
    ref_data = reference.mean_profile(use_mean=params.use_mean)
    partial_data = partial.mean_profile(use_mean=params.use_mean)

    partial_length = len(partial_data)

    # If no candidate positions provided, find them
    # Use a local list variable to track positions (avoids type narrowing issues)
    positions: list[int]
    if candidate_positions is None:
        candidate_positions_arr, _, comp_scale = find_match_candidates(
            reference, partial, params
        )
        positions = candidate_positions_arr.tolist()

        # Adjust scale passes to only use scales at or below comp_scale
        adjusted_passes = tuple(s for s in params.scale_passes if s <= comp_scale)
        params = AlignmentParameters(
            scale_passes=adjusted_passes if adjusted_passes else params.scale_passes,
            max_translation=params.max_translation,
            max_scaling=params.max_scaling,
            cutoff_hi=params.cutoff_hi,
            cutoff_lo=params.cutoff_lo,
            partial_mark_threshold=params.partial_mark_threshold,
            inclusion_threshold=params.inclusion_threshold,
            use_mean=params.use_mean,
            remove_boundary_zeros=params.remove_boundary_zeros,
            cut_borders_after_smoothing=params.cut_borders_after_smoothing,
        )
    else:
        positions = list(candidate_positions)

    # Handle empty candidates by lowering threshold until we find some
    # MATLAB: while label_opts_empty == 1: ... param.inclusion_threshold -= 0.05
    threshold = params.inclusion_threshold
    while len(positions) == 0 and threshold > 0:
        threshold -= 0.05
        adjusted_params = AlignmentParameters(
            scale_passes=params.scale_passes,
            max_translation=params.max_translation,
            max_scaling=params.max_scaling,
            cutoff_hi=params.cutoff_hi,
            cutoff_lo=params.cutoff_lo,
            partial_mark_threshold=params.partial_mark_threshold,
            inclusion_threshold=threshold,
            use_mean=params.use_mean,
            remove_boundary_zeros=params.remove_boundary_zeros,
            cut_borders_after_smoothing=params.cut_borders_after_smoothing,
        )
        candidate_positions_arr, _, _ = find_match_candidates(
            reference, partial, adjusted_params
        )
        positions = candidate_positions_arr.tolist()

    # If still no candidates, use the start
    if len(positions) == 0:
        positions = [0]

    # Try each candidate position
    best_correlation = -np.inf
    best_result: AlignmentResult | None = None
    best_start = positions[0]

    for candidate_start in positions:
        # Extract reference segment at this candidate position
        end_idx = min(candidate_start + partial_length, len(ref_data))
        ref_segment_data = ref_data[candidate_start:end_idx]

        # Create Profile object for the segment
        ref_segment = Profile(
            depth_data=ref_segment_data,
            pixel_size=reference.pixel_size,
            cutoff_hi=reference.cutoff_hi,
            cutoff_lo=reference.cutoff_lo,
            resolution_limit=reference.resolution_limit,
        )

        # Create partial Profile with potentially trimmed data if segment is shorter
        if len(ref_segment_data) < partial_length:
            partial_trimmed = Profile(
                depth_data=partial_data[: len(ref_segment_data)],
                pixel_size=partial.pixel_size,
                cutoff_hi=partial.cutoff_hi,
                cutoff_lo=partial.cutoff_lo,
                resolution_limit=partial.resolution_limit,
            )
        else:
            partial_trimmed = Profile(
                depth_data=partial_data,
                pixel_size=partial.pixel_size,
                cutoff_hi=partial.cutoff_hi,
                cutoff_lo=partial.cutoff_lo,
                resolution_limit=partial.resolution_limit,
            )

        # Run alignment
        try:
            result = align_profiles_multiscale(ref_segment, partial_trimmed, params)

            if result.final_correlation > best_correlation:
                best_correlation = result.final_correlation
                best_result = result
                best_start = candidate_start
        except ValueError:
            # Length mismatch - skip this candidate
            continue

    # If no valid result, create a default one
    if best_result is None:
        best_result = AlignmentResult(
            transforms=(TransformParameters(translation=0.0, scaling=1.0),),
            correlation_history=np.array([[np.nan, np.nan]]),
            final_correlation=np.nan,
            reference_aligned=ref_data,
            compared_aligned=partial_data,
            total_translation=0.0,
            total_scaling=1.0,
        )

    return best_result, best_start
