"""Candidate search for partial profile matching.

This module provides functions for finding potential alignment positions when
comparing a partial (shorter) profile to a longer reference profile. The search
uses a multi-scale brute-force approach to identify candidate positions where
the correlation exceeds a threshold.

The main function is:
- find_match_candidates: Brute-force search for partial profile match positions

All length parameters are in meters (SI units).
"""

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

from conversion.filter.gaussian import ALPHA_GAUSSIAN
from conversion.filter.regression import apply_order0_filter, create_gaussian_kernel_1d
from conversion.profile_correlator.data_types import AlignmentParameters, Profile
from conversion.profile_correlator.similarity import compute_cross_correlation


def _apply_lowpass_filter_1d(
    profile: NDArray[np.floating],
    cutoff_wavelength: float,
    pixel_size: float,
    cut_borders: bool = False,
) -> NDArray[np.floating]:
    """
    Apply Gaussian low-pass filter to a 1D profile with NaN handling.

    Uses the 2D filtering infrastructure from conversion/filter by reshaping
    the 1D profile to 2D, applying the filter, and reshaping back.

    :param profile: 1D array of heights. May contain NaN values.
    :param cutoff_wavelength: Filter cutoff wavelength in meters.
    :param pixel_size: Sample spacing in meters.
    :param cut_borders: If True, trim filter-affected borders.
    :returns: Low-pass filtered profile.
    """
    profile = np.asarray(profile).ravel()

    # Convert cutoff to pixels
    cutoff_pixels = cutoff_wavelength / pixel_size

    # Check for NaN values
    has_nans = np.any(np.isnan(profile))

    # Create 1D Gaussian kernel
    kernel_1d = create_gaussian_kernel_1d(cutoff_pixels, has_nans, ALPHA_GAUSSIAN)

    # Identity kernel for the other axis
    kernel_identity = np.array([1.0])

    # Reshape 1D to 2D (N, 1)
    data_2d = profile[:, np.newaxis]

    # Apply 2D filter with identity kernel for x-axis (filter along y-axis only)
    mode = "constant" if has_nans else "symmetric"
    filtered_2d = apply_order0_filter(data_2d, kernel_identity, kernel_1d, mode=mode)

    # Preserve NaN positions
    if has_nans:
        filtered_2d[np.isnan(data_2d)] = np.nan

    # Reshape back to 1D
    filtered = filtered_2d.ravel()

    # Optionally cut borders
    if cut_borders:
        sigma = ALPHA_GAUSSIAN * cutoff_pixels / np.sqrt(2 * np.pi)
        border = int(round(sigma))
        if border > 0 and len(filtered) > 2 * border:
            filtered = filtered[border:-border]

    return filtered


def _apply_highpass_filter_1d(
    profile: NDArray[np.floating],
    cutoff_wavelength: float,
    pixel_size: float,
    cut_borders: bool = False,
) -> NDArray[np.floating]:
    """
    Apply Gaussian high-pass filter to a 1D profile.

    High-pass = original - low-pass.

    :param profile: 1D array of heights. May contain NaN values.
    :param cutoff_wavelength: Filter cutoff wavelength in meters.
    :param pixel_size: Sample spacing in meters.
    :param cut_borders: If True, trim filter-affected borders.
    :returns: High-pass filtered profile.
    """
    profile = np.asarray(profile).ravel()

    # Compute low-pass filtered version
    lowpass = _apply_lowpass_filter_1d(
        profile, cutoff_wavelength, pixel_size, cut_borders=False
    )

    # High-pass = original - low-pass
    highpass = profile - lowpass

    # Optionally cut borders
    if cut_borders:
        cutoff_pixels = cutoff_wavelength / pixel_size
        sigma = ALPHA_GAUSSIAN * cutoff_pixels / np.sqrt(2 * np.pi)
        border = int(round(sigma))
        if border > 0 and len(highpass) > 2 * border:
            highpass = highpass[border:-border]

    return highpass


def _label_connected_regions(binary_array: NDArray[np.bool_]) -> NDArray[np.intp]:
    """
    Label connected regions in a 1D binary array.

    This function assigns unique integer labels to contiguous regions of True
    values in the input array. It replaces MATLAB's DIPimage label() function.

    :param binary_array: 1D boolean array.
    :returns: Array of same shape with integer labels (0 = background).
    """
    # Use scipy.ndimage.label for connected component labeling
    # For 1D, structure is just [1, 1, 1] for connectivity
    # Cast result to tuple to help pyright understand the return type
    label_result: tuple[NDArray[np.intp], int] = ndimage.label(  # type: ignore[assignment]
        binary_array.astype(int)
    )
    return np.asarray(label_result[0], dtype=np.intp)


def find_match_candidates(
    reference: Profile,
    partial: Profile,
    params: AlignmentParameters | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.floating], float]:
    """
    Find candidate positions for partial profile alignment using multi-scale search.

    This function performs a brute-force search to find positions in the reference
    profile where the partial profile might align. The search is done at multiple
    scales (cutoff wavelengths), and only positions that exceed the correlation
    threshold at all scales are considered candidates.

    The algorithm:
    1. Determine which scales to evaluate based on profile length and cutoffs
    2. At each "shape" scale, apply low-pass filter and compute sliding correlation
    3. Find positions where correlation exceeds threshold at all shape scales
    4. At the comparison scale, use high-pass filtered profiles to refine positions
    5. Return the best position within each candidate region

    All length parameters are in meters (SI units).

    This corresponds to MATLAB's DetermineMatchCandidatesMultiScale.m.

    :param reference: Reference profile (longer one).
    :param partial: Partial profile (shorter one) to find in reference.
    :param params: Alignment parameters. If None, default parameters are used.
    :returns: Tuple of (candidate_positions, shape_scales, comparison_scale) where:
        - candidate_positions: Array of starting indices in the reference
        - shape_scales: Array of scale values used for shape filtering (in meters)
        - comparison_scale: Scale value used for final comparison (in meters)

    Example::

        >>> import numpy as np
        >>> from conversion.profile_correlator.data_types import Profile
        >>> # Create reference with a matching segment
        >>> x = np.linspace(0, 20, 2000)
        >>> ref = Profile(np.sin(x), pixel_size=0.5e-6)
        >>> partial = Profile(np.sin(x[500:1000]), pixel_size=0.5e-6)
        >>> positions, shape_scales, comp_scale = find_match_candidates(ref, partial)
        >>> len(positions) > 0
        True
    """
    # Use default parameters if not provided
    if params is None:
        params = AlignmentParameters()

    # Extract data
    ref_data = reference.mean_profile(use_mean=params.use_mean)
    partial_data = partial.mean_profile(use_mean=params.use_mean)

    # Get physical parameters (in meters)
    pixel_size = reference.pixel_size

    # Determine effective cutoff bounds (all in meters)
    cutoff_hi = params.cutoff_hi
    cutoff_lo = params.cutoff_lo

    if reference.cutoff_hi is not None and partial.cutoff_hi is not None:
        cutoff_hi = min(cutoff_hi, reference.cutoff_hi, partial.cutoff_hi)
    if reference.cutoff_lo is not None and partial.cutoff_lo is not None:
        cutoff_lo = max(cutoff_lo, reference.cutoff_lo, partial.cutoff_lo)

    # Get resolution limit (in meters)
    if reference.resolution_limit is not None:
        resolution_limit = reference.resolution_limit
    else:
        resolution_limit = max(cutoff_lo, 2 * pixel_size)

    # Define evaluation scales (all in meters from params.scale_passes)
    possible_scales = np.array(list(params.scale_passes), dtype=np.float64)

    # Filter scales by cutoff bounds
    possible_scales = possible_scales[possible_scales <= cutoff_hi]
    possible_scales = possible_scales[possible_scales >= resolution_limit]

    # Maximum scale is based on partial profile length (in meters)
    max_scale = pixel_size * len(partial_data) / 2

    # Filter to scales <= 4 * max_scale
    possible_scales = possible_scales[possible_scales <= 4 * max_scale]

    if len(possible_scales) == 0:
        # No valid scales - return single candidate at position 0
        return np.array([0], dtype=np.intp), np.array([]), cutoff_lo

    # Determine shape scales vs comparison scale
    # Shape scales are >= max_scale, comparison scale is the first one < max_scale
    eval_scales = possible_scales < max_scale

    shape_scale_indices = np.where(~eval_scales)[0]
    comp_scale_indices = np.where(eval_scales)[0]

    if len(comp_scale_indices) == 0:
        # All scales are shape scales - use the finest one for comparison
        comp_scale_idx = len(possible_scales) - 1
        shape_scale_indices = np.arange(len(possible_scales) - 1)
    else:
        comp_scale_idx = comp_scale_indices[0]

    shape_scales = possible_scales[shape_scale_indices]
    comp_scale = possible_scales[comp_scale_idx]

    # Oversampling factor for subsampling
    oversampling = 16

    # If we have shape scales, do multi-scale correlation search
    if len(shape_scales) > 0:
        # Generate filtered profiles at each shape scale and compute correlations
        xcorr_by_scale: list[NDArray[np.floating]] = []
        array_lengths: list[int] = []

        for scale in shape_scales:
            # Low-pass filter both profiles (all in meters)
            ref_filtered = _apply_lowpass_filter_1d(
                ref_data, scale, pixel_size, cut_borders=False
            )
            partial_filtered = _apply_lowpass_filter_1d(
                partial_data, scale, pixel_size, cut_borders=False
            )

            # Compute subsampling factor based on scale in samples
            scale_samples = scale / pixel_size
            subsampling = max(1, int(np.floor(scale_samples / oversampling)))

            # Resample for efficiency
            ref_length_sub = max(1, int(round(len(ref_filtered) / subsampling)))
            partial_length_sub = max(1, int(round(len(partial_filtered) / subsampling)))

            # Cast resample results to float64 arrays
            ref_filtered_sub: NDArray[np.floating] = np.asarray(
                signal.resample(ref_filtered, ref_length_sub), dtype=np.float64
            )
            partial_filtered_sub: NDArray[np.floating] = np.asarray(
                signal.resample(partial_filtered, partial_length_sub), dtype=np.float64
            )

            # Compute correlation at each position by sliding partial along reference
            n_positions = max(1, len(ref_filtered_sub) - len(partial_filtered_sub) + 1)
            xcorr_tmp = np.zeros(n_positions)

            for pos in range(n_positions):
                end_pos = pos + len(partial_filtered_sub)
                ref_segment = ref_filtered_sub[pos:end_pos]

                # Ensure same length
                min_len = min(len(ref_segment), len(partial_filtered_sub))
                if min_len > 0:
                    xcorr_tmp[pos] = compute_cross_correlation(
                        ref_segment[:min_len], partial_filtered_sub[:min_len]
                    )
                else:
                    xcorr_tmp[pos] = np.nan

            # Upsample back to original resolution
            target_length = max(1, len(ref_data) - len(partial_data) + 1)
            xcorr_upsampled: NDArray[np.floating] = np.asarray(
                signal.resample(xcorr_tmp, target_length), dtype=np.float64
            )

            xcorr_by_scale.append(xcorr_upsampled)
            array_lengths.append(len(xcorr_upsampled))

        # Stack correlations and find positions above threshold at all scales
        min_length = min(array_lengths)
        xcorr_array = np.vstack([x[:min_length] for x in xcorr_by_scale])

        # Threshold and find candidates
        xcorr_thresh = xcorr_array >= params.inclusion_threshold
        candidates = np.sum(xcorr_thresh, axis=0) == len(shape_scales)

        # Label connected candidate regions
        candidates_labeled = _label_connected_regions(candidates)

    else:
        # No shape scales - all positions are candidates
        n_positions = max(1, len(ref_data) - len(partial_data) + 1)
        candidates_labeled = np.ones(n_positions, dtype=np.intp)

    # Refine candidates at comparison scale using high-pass filtered profiles
    ref_filtered_comp = _apply_highpass_filter_1d(
        ref_data, comp_scale, pixel_size, cut_borders=False
    )
    partial_filtered_comp = _apply_highpass_filter_1d(
        partial_data, comp_scale, pixel_size, cut_borders=False
    )

    # For each labeled region, find the best position
    label_opts: list[int] = []
    n_labels = int(np.max(candidates_labeled))

    for label_id in range(1, n_labels + 1):
        # Get positions with this label
        label_positions = np.where(candidates_labeled == label_id)[0]

        # Filter to valid positions
        max_pos = len(ref_filtered_comp) - len(partial_filtered_comp) + 1
        label_positions = label_positions[label_positions < max_pos]

        if len(label_positions) == 0:
            continue

        # Find best position within this region
        best_xcorr = -np.inf
        best_pos = label_positions[0]

        for pos in range(label_positions[0], label_positions[-1] + 1):
            if pos >= max_pos:
                break

            end_pos = pos + len(partial_filtered_comp)
            ref_segment = ref_filtered_comp[pos:end_pos]

            # Compute correlation
            min_len = min(len(ref_segment), len(partial_filtered_comp))
            if min_len > 0:
                xcorr = compute_cross_correlation(
                    ref_segment[:min_len], partial_filtered_comp[:min_len]
                )
                if not np.isnan(xcorr) and xcorr > best_xcorr:
                    best_xcorr = xcorr
                    best_pos = pos

        label_opts.append(best_pos)

    # Convert to arrays
    candidate_positions = np.array(label_opts, dtype=np.intp)

    return candidate_positions, shape_scales, comp_scale
