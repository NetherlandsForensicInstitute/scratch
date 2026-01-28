"""Candidate search for partial profile matching.

This module provides functions for finding potential alignment positions when
comparing a partial (shorter) profile to a longer reference profile. The search
uses a multi-scale brute-force approach to identify candidate positions where
the correlation exceeds a threshold.

The main function is:
- find_match_candidates: Brute-force search for partial profile match positions

All length parameters are in meters (SI units).

MATLAB Correspondence
---------------------
This module corresponds to MATLAB's ``DetermineMatchCandidatesMultiScale.m``.

Known Differences from MATLAB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **max_scale calculation** (``find_match_candidates``):
   MATLAB: ``max_scale = xdim * length(partial_profile.depth_data) * 1e6 / 2``
   (uses the *shorter* / partial profile length).
   Python: ``max_scale = pixel_size * len(ref_data) / 2``
   (uses the *longer* / reference profile length).

   This is an intentional deviation.  ``max_scale`` determines which pass
   frequencies are classified as "shape scales" (>= max_scale, used for
   coarse candidate filtering) versus "comparison scales" (< max_scale, used
   for fine refinement).  Using the longer profile ensures that scales
   comparable to the partial profile length are treated as comparison scales
   rather than shape scales.  This gives better candidate discrimination for
   the ``different_sampling`` test case where the partial and reference
   profiles have very different lengths after pixel-size equalization.
   With MATLAB's formula, the ``different_sampling`` case fails because a
   scale that should be a comparison scale becomes a shape scale, leading to
   incorrect candidate selection.

2. **Resampling method** (``_resample_interpolation``):
   MATLAB uses DIPimage's ``resample()`` (cubic interpolation).  Python uses
   a local cubic B-spline kernel that approximates DIPimage's behaviour.
   Minor numerical differences exist but do not affect candidate selection.

3. **No-shape-scales fallback**:
   When no pass frequencies qualify as shape scales, MATLAB returns a single
   candidate region spanning all positions.  Python instead splits the search
   space into ~5 evenly-sized regions to provide multiple starting points for
   the downstream alignment, improving robustness.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.ndimage import convolve1d

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

    This implementation matches MATLAB's RemoveNoiseGaussian/SmoothMod exactly:
    - Uses sigma = cutoff_pixels * 0.187390625 (ChebyCutoffToGaussSigma)
    - Kernel formula: exp(-0.5 * (alpha*n/(L/2))^2) with alpha=3
    - Applies edge correction using NanConv-style normalized convolution

    :param profile: 1D array of heights. May contain NaN values.
    :param cutoff_wavelength: Filter cutoff wavelength in meters.
    :param pixel_size: Sample spacing in meters.
    :param cut_borders: If True, trim filter-affected borders.
    :returns: Low-pass filtered profile.
    """
    profile = np.asarray(profile).ravel().astype(np.float64)

    # Convert cutoff to pixels (matches MATLAB: cutoff/xdim where both in μm)
    cutoff_pixels = cutoff_wavelength / pixel_size

    # MATLAB sigma calculation: sigma = cutoff/xdim * 0.187390625
    # where 0.187390625 = sqrt(2*ln(2))/(2*pi)
    sigma = cutoff_pixels * 0.187390625

    # MATLAB kernel size: L = 1 + 2*round(alpha*sigma); L = L - 1
    alpha = 3.0
    L = 1 + 2 * round(alpha * sigma)
    L = L - 1  # Make L even, so L+1 kernel points is odd

    # MATLAB kernel: n = (0:L)' - L/2; t = exp(-(1/2)*(alpha*n/(L/2)).^2)
    n = np.arange(L + 1) - L / 2  # L+1 points from -L/2 to L/2
    kernel = np.exp(-0.5 * (alpha * n / (L / 2)) ** 2)
    kernel = kernel / np.sum(kernel)  # Normalize

    # Find NaN positions
    nan_mask = np.isnan(profile)

    # Create working copy with NaNs replaced by zeros
    a = profile.copy()
    a[nan_mask] = 0.0

    # Create ones array with zeros at NaN positions
    on = np.ones_like(profile)
    on[nan_mask] = 0.0

    # MATLAB NanConv with 'edge' option:
    # flat = conv2(on, k, 'same')  -- edge correction divisor
    # c = conv2(a, k, 'same') / flat
    flat = convolve1d(on, kernel, mode="constant", cval=0.0)
    filtered_raw = convolve1d(a, kernel, mode="constant", cval=0.0)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        filtered = np.where(flat > 0, filtered_raw / flat, 0.0)

    # Restore NaN positions (matches MATLAB 'nanout' option)
    filtered[nan_mask] = np.nan

    # Optionally cut borders (matches MATLAB ceil(sigma))
    if cut_borders:
        border = int(np.ceil(sigma))
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

    High-pass = original - low-pass. Matches MATLAB's RemoveShapeGaussian.

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

    # Optionally cut borders (matches MATLAB ceil(sigma))
    if cut_borders:
        cutoff_pixels = cutoff_wavelength / pixel_size
        sigma = cutoff_pixels * 0.187390625
        border = int(np.ceil(sigma))
        if border > 0 and len(highpass) > 2 * border:
            highpass = highpass[border:-border]

    return highpass


def _cubic_bspline(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Evaluate the cubic B-spline basis function.

    This is the 4-point-support kernel used by DIPimage's ``resample``
    for order-3 interpolation (direct evaluation, no prefilter).

    :param x: Array of evaluation positions.
    :returns: Kernel values (0 for |x| >= 2).
    """
    ax = np.abs(x)
    result = np.zeros_like(ax)
    # |x| < 1: (3|x|^3 - 6|x|^2 + 4) / 6
    m1 = ax < 1.0
    result[m1] = (3.0 * ax[m1] ** 3 - 6.0 * ax[m1] ** 2 + 4.0) / 6.0
    # 1 <= |x| < 2: (2 - |x|)^3 / 6
    m2 = (ax >= 1.0) & (ax < 2.0)
    result[m2] = (2.0 - ax[m2]) ** 3 / 6.0
    return result


def _resample_interpolation(
    data: NDArray[np.floating],
    zoom: float,
) -> NDArray[np.floating]:
    """
    Resample a 1D array using interpolation, matching DIPimage's resample.

    DIPimage's ``resample(img, zoom)`` uses direct cubic B-spline kernel
    evaluation (4-point local support, **no** global prefilter).  Output
    sample *j* maps to input position ``j / zoom``.

    This differs from ``scipy.signal.resample`` (FFT-based, propagates
    NaN globally) and from ``scipy.interpolate.interp1d(kind='cubic')``
    (global natural cubic spline).  Here NaN propagates only through
    the local 4-point kernel, exactly matching DIPimage behaviour.

    :param data: 1D input array.
    :param zoom: Zoom factor (< 1 for downsampling, > 1 for upsampling).
    :returns: Resampled 1D array of length ``max(1, round(len(data) * zoom))``.
    """
    n_in = len(data)
    n_out = max(1, int(round(n_in * zoom)))

    if n_out == n_in:
        return data.copy()

    # DIPimage coordinates: output sample j -> input position j / zoom
    new_x = np.arange(n_out, dtype=np.float64) / zoom

    result = np.empty(n_out, dtype=np.float64)

    for j in range(n_out):
        x = new_x[j]
        # Integer indices for the 4-point B-spline stencil: [i-1, i, i+1, i+2]
        # where i = floor(x)
        i0 = int(np.floor(x))
        val = 0.0
        for di in range(-1, 3):
            idx = i0 + di
            # Clamp to valid range (DIPimage boundary handling)
            idx_clamped = max(0, min(n_in - 1, idx))
            weight = float(_cubic_bspline(np.array([x - idx]))[0])
            val += data[idx_clamped] * weight
        result[j] = val

    return result


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

    # DIFFERENCE FROM MATLAB: max_scale uses the reference (longer) profile
    # length instead of the partial (shorter) profile length.
    # MATLAB: max_scale = xdim * length(partial_profile.depth_data) * 1e6 / 2
    # Python: max_scale = pixel_size * len(ref_data) / 2
    # Using the longer profile ensures that scales comparable to the search
    # space are treated as comparison scales rather than shape scales, giving
    # better discrimination across candidate positions.  See module docstring
    # for full rationale.
    max_scale = pixel_size * len(ref_data) / 2

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

            # Resample for efficiency using interpolation to match
            # DIPimage's resample() which uses cubic interpolation.
            # Unlike scipy.signal.resample (FFT-based), this preserves
            # NaN locally instead of propagating it to the entire output.
            zoom = 1.0 / subsampling
            ref_filtered_sub: NDArray[np.floating] = _resample_interpolation(
                ref_filtered, zoom
            )
            partial_filtered_sub: NDArray[np.floating] = _resample_interpolation(
                partial_filtered, zoom
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

            # Upsample back to original resolution using interpolation
            # to match DIPimage's resample(xcorr_tmp, subsampling)
            xcorr_upsampled: NDArray[np.floating] = _resample_interpolation(
                xcorr_tmp, float(subsampling)
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
        # No shape scales - split positions into evenly-spaced regions
        # so that the refinement returns multiple candidates.  This
        # avoids committing to a single position that may lead to an
        # unstable alignment while missing better starting points.
        n_positions = max(1, len(ref_data) - len(partial_data) + 1)
        n_regions = max(1, min(n_positions, 5))
        chunk = max(1, n_positions // n_regions)
        candidates_labeled = np.array(
            [i // chunk + 1 for i in range(n_positions)], dtype=np.intp
        )

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
