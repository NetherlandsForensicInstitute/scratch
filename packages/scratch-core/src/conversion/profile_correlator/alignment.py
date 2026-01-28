"""
Multi-scale profile alignment algorithms.

This module provides functions for aligning 1D profiles using multi-scale
coarse-to-fine registration. The algorithm iteratively refines the alignment
at progressively finer scales to achieve accurate registration while avoiding
local minima.

The main functions are:
- align_profiles_multiscale: Full profile multi-scale alignment
- align_partial_profile_multiscale: Partial profile alignment with candidate search

All length parameters are in meters (SI units).
"""

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    AlignmentResult,
    Profile,
    TransformParameters,
)
from conversion.profile_correlator.similarity import compute_cross_correlation
from conversion.profile_correlator.transforms import (
    apply_transform,
    compute_cumulative_transform,
)


def _fminsearchbnd_transform_to_unconstrained(
    x: NDArray[np.floating],
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Transform bounded variables to unconstrained space using MATLAB fminsearchbnd approach.

    For doubly-bounded variables: uses sin transformation
    This matches MATLAB's fminsearchbnd.m exactly.

    :param x: Variables in bounded space.
    :param lb: Lower bounds.
    :param ub: Upper bounds.
    :returns: Variables in unconstrained space.
    """
    xu = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        if np.isfinite(lb[i]) and np.isfinite(ub[i]):
            # Doubly bounded - use sin transformation (case 3 in MATLAB)
            if x[i] <= lb[i]:
                # Infeasible starting value
                xu[i] = -np.pi / 2
            elif x[i] >= ub[i]:
                # Infeasible starting value
                xu[i] = np.pi / 2
            else:
                # Normalize to [-1, 1]
                normalized = 2 * (x[i] - lb[i]) / (ub[i] - lb[i]) - 1
                normalized = max(-1.0, min(1.0, normalized))
                # Shift by 2*pi to avoid problems at zero in fminsearch
                # "otherwise, the initial simplex is vanishingly small"
                xu[i] = 2 * np.pi + np.arcsin(normalized)
        else:
            # Unconstrained (case 0 in MATLAB)
            xu[i] = x[i]
    return xu


def _fminsearchbnd_transform_to_bounded(
    xu: NDArray[np.floating],
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Transform unconstrained variables back to bounded space using MATLAB fminsearchbnd approach.

    For doubly-bounded variables: uses sin transformation
    This matches MATLAB's fminsearchbnd.m xtransform function exactly.

    :param xu: Variables in unconstrained space.
    :param lb: Lower bounds.
    :param ub: Upper bounds.
    :returns: Variables in bounded space.
    """
    x = np.zeros_like(xu, dtype=np.float64)
    for i in range(len(xu)):
        if np.isfinite(lb[i]) and np.isfinite(ub[i]):
            # Doubly bounded - use sin transformation (case 3 in MATLAB)
            # xtrans(i) = (sin(x(k))+1)/2;
            # xtrans(i) = xtrans(i)*(params.UB(i) - params.LB(i)) + params.LB(i);
            x[i] = (np.sin(xu[i]) + 1) / 2
            x[i] = x[i] * (ub[i] - lb[i]) + lb[i]
            # Just in case of any floating point problems
            x[i] = max(lb[i], min(ub[i], x[i]))
        else:
            # Unconstrained (case 0 in MATLAB)
            x[i] = xu[i]
    return x


def _fminsearchbnd_objective_wrapper(
    xu: NDArray[np.floating],
    objective_func,
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
    *args,
) -> float:
    """
    Wrapper for objective function that transforms variables from unconstrained to bounded space.

    This is used to implement MATLAB's fminsearchbnd approach where the optimizer
    works in unconstrained space but the objective is evaluated in bounded space.

    :param xu: Variables in unconstrained space.
    :param objective_func: The actual objective function that expects bounded variables.
    :param lb: Lower bounds.
    :param ub: Upper bounds.
    :param args: Additional arguments to pass to objective_func.
    :returns: Objective function value.
    """
    x = _fminsearchbnd_transform_to_bounded(xu, lb, ub)
    return objective_func(x, *args)


def _matlab_fminsearch(
    fun: Callable[..., float],
    x0: NDArray[np.floating],
    tol_x: float = 1e-4,
    tol_fun: float = 1e-4,
    max_iter: int = 400,
    max_fun_evals: int = 400,
    args: tuple = (),
) -> NDArray[np.floating]:
    """
    MATLAB-exact implementation of fminsearch (Nelder-Mead simplex method).

    This replicates MATLAB's fminsearch.m algorithm exactly, including:
    - Initial simplex creation (5% perturbation, 0.00025 for zero elements)
    - Reflection, expansion, contraction, and shrink operations
    - Termination criteria matching MATLAB
    - Coefficient values: rho=1, chi=2, psi=0.5, sigma=0.5

    :param fun: Objective function to minimize.
    :param x0: Initial point (1D array).
    :param tol_x: Termination tolerance on x (MATLAB TolX).
    :param tol_fun: Termination tolerance on function value (MATLAB TolFun).
    :param max_iter: Maximum number of iterations (MATLAB MaxIter).
    :param max_fun_evals: Maximum function evaluations (MATLAB MaxFunEvals).
    :param args: Additional arguments passed to fun.
    :returns: Optimized point.
    """
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    n = len(x0)

    # Nelder-Mead coefficients (matching MATLAB exactly)
    rho = 1.0  # reflection
    chi = 2.0  # expansion
    psi = 0.5  # contraction
    sigma = 0.5  # shrink

    # Build initial simplex (matching MATLAB's fminsearch.m)
    usual_delta = 0.05
    zero_term_delta = 0.00025

    # v[i] is vertex i, v has n+1 rows (vertices) x n columns (dimensions)
    v = np.zeros((n + 1, n), dtype=np.float64)
    v[0] = x0.copy()
    for j in range(n):
        y = x0.copy()
        if y[j] != 0:
            y[j] = (1 + usual_delta) * y[j]
        else:
            y[j] = zero_term_delta
        v[j + 1] = y

    # Evaluate function at all vertices
    fv = np.zeros(n + 1, dtype=np.float64)
    for i in range(n + 1):
        fv[i] = fun(v[i], *args)
    func_evals = n + 1

    # Sort vertices by function value (stable sort to match MATLAB's sort)
    sort_idx = np.argsort(fv, kind="stable")
    fv = fv[sort_idx]
    v = v[sort_idx]

    iterations = 0

    while True:
        # Check convergence (matching MATLAB's criteria)
        # max(abs(fv(1) - fv(2:end))) <= max(tolfun, 10*eps(fv(1)))
        # max(max(abs(v(2:end,:) - v(1,:)))) <= max(tolx, 10*eps(max(v(1,:))))
        fv_diff = np.max(np.abs(fv[0] - fv[1:]))
        v_diff = np.max(np.abs(v[1:] - v[0]))

        if fv_diff <= max(
            tol_fun, 10 * np.finfo(np.float64).eps * abs(fv[0])
        ) and v_diff <= max(
            tol_x, 10 * np.finfo(np.float64).eps * np.max(np.abs(v[0]))
        ):
            break

        if iterations >= max_iter:
            break
        if func_evals >= max_fun_evals:
            break

        iterations += 1

        # Compute centroid of all vertices except the worst
        xbar = np.mean(v[:n], axis=0)

        # Reflection
        xr = (1 + rho) * xbar - rho * v[n]
        fxr = fun(xr, *args)
        func_evals += 1

        if fxr < fv[0]:
            # Reflected point is better than best - try expansion
            xe = (1 + rho * chi) * xbar - rho * chi * v[n]
            fxe = fun(xe, *args)
            func_evals += 1

            if fxe < fxr:
                # Expansion is better
                v[n] = xe
                fv[n] = fxe
            else:
                # Reflection is better than expansion
                v[n] = xr
                fv[n] = fxr

        elif fxr < fv[n - 1]:
            # Reflected is better than second-worst - accept reflection
            v[n] = xr
            fv[n] = fxr

        else:
            # Need to contract
            if fxr < fv[n]:
                # Outside contraction
                xc = (1 + psi * rho) * xbar - psi * rho * v[n]
                fxc = fun(xc, *args)
                func_evals += 1

                if fxc <= fxr:
                    v[n] = xc
                    fv[n] = fxc
                else:
                    # Shrink
                    for j in range(1, n + 1):
                        v[j] = v[0] + sigma * (v[j] - v[0])
                        fv[j] = fun(v[j], *args)
                    func_evals += n
            else:
                # Inside contraction
                xcc = (1 - psi) * xbar + psi * v[n]
                fxcc = fun(xcc, *args)
                func_evals += 1

                if fxcc < fv[n]:
                    v[n] = xcc
                    fv[n] = fxcc
                else:
                    # Shrink
                    for j in range(1, n + 1):
                        v[j] = v[0] + sigma * (v[j] - v[0])
                        fv[j] = fun(v[j], *args)
                    func_evals += n

        # Sort vertices by function value (stable sort to match MATLAB's sort)
        sort_idx = np.argsort(fv, kind="stable")
        fv = fv[sort_idx]
        v = v[sort_idx]

    return v[0]


def _fminsearchbnd(
    fun: Callable[..., float],
    x0: NDArray[np.floating],
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
    tol_x: float = 1e-4,
    tol_fun: float = 1e-4,
    max_iter: int = 400,
    max_fun_evals: int = 400,
    args: tuple = (),
) -> NDArray[np.floating]:
    """
    MATLAB-exact implementation of fminsearchbnd (bounded Nelder-Mead).

    Wraps _matlab_fminsearch with sin transformation for bounded variables,
    exactly matching MATLAB's fminsearchbnd.m.

    :param fun: Objective function to minimize.
    :param x0: Initial point in bounded space.
    :param lb: Lower bounds.
    :param ub: Upper bounds.
    :param tol_x: Termination tolerance on x.
    :param tol_fun: Termination tolerance on function value.
    :param max_iter: Maximum iterations.
    :param max_fun_evals: Maximum function evaluations.
    :param args: Additional arguments passed to fun.
    :returns: Optimized point in bounded space.
    """
    x0u = _fminsearchbnd_transform_to_unconstrained(x0, lb, ub)

    def wrapped_fun(xu: NDArray[np.floating], *extra_args: object) -> float:
        x_bounded = _fminsearchbnd_transform_to_bounded(xu, lb, ub)
        return fun(x_bounded, *extra_args)

    xu_opt = _matlab_fminsearch(
        wrapped_fun,
        x0u,
        tol_x=tol_x,
        tol_fun=tol_fun,
        max_iter=max_iter,
        max_fun_evals=max_fun_evals,
        args=args,
    )
    return _fminsearchbnd_transform_to_bounded(xu_opt, lb, ub)


def _apply_lowpass_filter_1d(
    profile: NDArray[np.floating],
    cutoff_wavelength: float,
    pixel_size: float,
    cut_borders: bool = False,
) -> NDArray[np.floating]:
    """
    Apply Gaussian low-pass filter to a 1D profile with NaN handling.

    This implementation matches MATLAB's ApplyLowPassFilter exactly:
    - Uses sigma = cutoff_pixels * 0.187390625 (sqrt(2*ln(2))/(2*pi))
    - Kernel formula: exp(-0.5 * (alpha*n/(L/2))^2) with alpha=3
    - Applies edge correction using NanConv-style normalized convolution

    :param profile: 1D array of heights. May contain NaN values.
    :param cutoff_wavelength: Filter cutoff wavelength in meters.
    :param pixel_size: Sample spacing in meters.
    :param cut_borders: If True, trim filter-affected borders.
    :returns: Low-pass filtered profile.
    """
    from scipy.ndimage import convolve1d

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
    has_nans = np.any(nan_mask)

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

    # Optionally cut borders (matches MATLAB cut_borders_after_smoothing)
    if cut_borders:
        border = int(round(sigma))
        if border > 0 and len(filtered) > 2 * border:
            filtered = filtered[border:-border]

    return filtered


def _remove_boundary_zeros(
    data_1: NDArray[np.floating],
    data_2: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """
    Remove zero-padded boundaries from two 1D profiles.

    Finds the common non-zero region in both profiles and crops them.

    :param data_1: First profile data.
    :param data_2: Second profile data.
    :returns: Tuple of (cropped_1, cropped_2, start_position).
    """
    # Create zero masks
    zero_mask_1 = data_1 == 0
    zero_mask_2 = data_2 == 0

    # Find non-zero indices for each profile
    nonzero_1 = np.where(~zero_mask_1)[0]
    nonzero_2 = np.where(~zero_mask_2)[0]

    # Handle empty cases
    if len(nonzero_1) == 0 or len(nonzero_2) == 0:
        return data_1[0:0], data_2[0:0], 0

    # Find bounds for each profile
    start_1, end_1 = nonzero_1[0], nonzero_1[-1] + 1
    start_2, end_2 = nonzero_2[0], nonzero_2[-1] + 1

    # Find common region
    start = max(start_1, start_2)
    end = min(end_1, end_2)

    if end <= start:
        return data_1[0:0], data_2[0:0], start

    return data_1[start:end], data_2[start:end], start


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

    :param x: Optimization variables [translation, encoded_scaling] where
        encoded_scaling = (scaling - 1) * 10000.
    :param profile_ref: Reference profile (1D array).
    :param profile_comp: Compared profile to be transformed (1D array).
    :returns: Negative cross-correlation (minimized during optimization).
    """
    # Extract and decode parameters
    translation = x[0]
    scaling = x[1] / 10000.0 + 1.0

    # Create transform and apply to compared profile
    # Wrap array in Profile for apply_transform (pixel_size=1.0 since we're in sample space)
    transform = TransformParameters(translation=translation, scaling=scaling)
    profile_comp_profile = Profile(depth_data=profile_comp, pixel_size=1.0)
    profile_comp_transformed = apply_transform(profile_comp_profile, transform)

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

    All length parameters are in meters (SI units).

    :param profile_ref: Reference profile (kept fixed during alignment).
    :param profile_comp: Compared profile (transformed to align with reference).
    :param params: Alignment parameters. If None, default parameters are used.
    :returns: AlignmentResult containing the sequence of transforms, correlation
        history, final correlation, and the aligned profiles.
    :raises ValueError: If profiles have different lengths. Use
        make_profiles_equal_length() first if needed.
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

    # Get pixel size in meters
    pixel_size = profile_ref.pixel_size

    # Determine resolution limit
    if profile_ref.resolution_limit is not None:
        resolution_limit = profile_ref.resolution_limit
    else:
        resolution_limit = max(params.cutoff_lo, 2 * pixel_size)

    # Determine effective cutoff bounds (all in meters)
    cutoff_hi = params.cutoff_hi
    cutoff_lo = params.cutoff_lo

    if profile_ref.cutoff_hi is not None and profile_comp.cutoff_hi is not None:
        cutoff_hi = min(cutoff_hi, profile_ref.cutoff_hi, profile_comp.cutoff_hi)
    if profile_ref.cutoff_lo is not None and profile_comp.cutoff_lo is not None:
        cutoff_lo = max(cutoff_lo, profile_ref.cutoff_lo, profile_comp.cutoff_lo)

    # Convert max_translation from meters to samples
    max_translation_samples = params.max_translation / pixel_size

    # MATLAB's redetermine_max_trans logic:
    # When max_translation is the default value (~10m = 1e7 μm), MATLAB sets
    # redetermine_max_trans=1 and uses the current pass value as the
    # translation limit at each scale level. This dramatically reduces
    # the search space and is crucial for correct convergence.
    # In MATLAB: max_translation(mm) == 10000 triggers this.
    max_translation_mm = params.max_translation * 1000  # meters to mm
    redetermine_max_trans = abs(max_translation_mm - 10000) < 1e-6

    if not redetermine_max_trans:
        # Non-default: convert to samples (MATLAB: round(max_translation/xdim/1000))
        max_translation_samples = round(max_translation_samples)

    # Initialize tracking variables
    transforms: list[TransformParameters] = []
    correlation_history: list[tuple[float, float]] = []

    # Track cumulative transform applied to profile_2
    translation_total = 0.0
    scaling_total = 1.0
    current_scaling = 1.0  # For adjusting bounds at each scale

    # Working copy of compared profile (recomputed from original each iteration)
    profile_2_mod = profile_2.copy()

    # Process each scale level from coarse to fine (scale_passes are in meters)
    # Use a small relative tolerance for comparisons to handle floating-point
    # imprecision when pass values are computed from integer μm values
    # (e.g. 10 * 1e-6 != 1e-5 due to IEEE 754 rounding).
    _ftol = 1e-9
    for cutoff in params.scale_passes:
        # Check if this scale level should be processed
        if cutoff < resolution_limit * (1 - _ftol):
            # Scale is finer than resolution limit, skip
            continue
        if cutoff > cutoff_hi * (1 + _ftol):
            # Scale is coarser than high cutoff bound, skip
            continue
        if cutoff < cutoff_lo * (1 - _ftol):
            # Scale is finer than low cutoff bound, skip
            continue

        # Apply low-pass filter to both profiles at current scale
        profile_1_filtered = _apply_lowpass_filter_1d(
            profile_1,
            cutoff_wavelength=cutoff,
            pixel_size=pixel_size,
            cut_borders=params.cut_borders_after_smoothing,
        )

        profile_2_filtered = _apply_lowpass_filter_1d(
            profile_2_mod,
            cutoff_wavelength=cutoff,
            pixel_size=pixel_size,
            cut_borders=params.cut_borders_after_smoothing,
        )

        # Compute subsampling factor for efficiency
        # Factor is based on cutoff wavelength in samples
        cutoff_samples = cutoff / pixel_size
        subsample_factor = max(1, int(np.ceil(cutoff_samples / 2 / 5)))

        # Subsample for optimization (coarse scales)
        profile_1_subsampled = profile_1_filtered[::subsample_factor]
        profile_2_subsampled = profile_2_filtered[::subsample_factor]

        # Compute bounds for this scale level
        max_trans_adj = max_translation_samples - translation_total
        min_trans_adj = max_translation_samples + translation_total
        max_scaling_adj = params.max_scaling * (1 - (scaling_total - 1))
        min_scaling_adj = params.max_scaling * (1 + (scaling_total - 1))

        # MATLAB redetermine_max_trans: override translation bounds with
        # current pass value (in μm, treated as sample-like units)
        if redetermine_max_trans:
            cutoff_um = cutoff * 1e6  # Convert meters to μm
            min_trans_adj = cutoff_um
            max_trans_adj = cutoff_um

        # Bounds in subsampled coordinates
        trans_lb = -int(round(min_trans_adj / subsample_factor))
        trans_ub = int(round(max_trans_adj / subsample_factor))

        # Note: the scaling bounds account for current_scaling
        scale_lb = ((1 - min_scaling_adj) / current_scaling - 1) * 10000
        scale_ub = ((1 + max_scaling_adj) / current_scaling - 1) * 10000

        # Bounds arrays
        lb = np.array([trans_lb, scale_lb], dtype=np.float64)
        ub = np.array([trans_ub, scale_ub], dtype=np.float64)

        # Initial guess: start from zero (matches MATLAB x0 = [0 0])
        x0 = np.array([0.0, 0.0])

        # Use MATLAB-exact fminsearchbnd implementation
        x_opt = _fminsearchbnd(
            _alignment_objective,
            x0,
            lb,
            ub,
            tol_x=1e-6,
            tol_fun=1e-6,
            max_iter=400,
            max_fun_evals=400,
            args=(profile_1_subsampled, profile_2_subsampled),
        )

        translation_subsampled = x_opt[0]
        scaling_encoded = x_opt[1]

        # Convert back to full resolution
        translation = translation_subsampled * subsample_factor
        scaling = scaling_encoded / 10000.0 + 1.0

        # Store transform for this scale level
        transform = TransformParameters(translation=translation, scaling=scaling)
        transforms.append(transform)

        # Update cumulative transform tracking
        current_scaling = scaling
        translation_total = translation_total + translation
        scaling_total = scaling_total * scaling

        # MATLAB recomputes profiles2_mod from the ORIGINAL profiles2 each iteration
        # by applying ALL accumulated transforms at once (single interpolation).
        # This avoids accumulating interpolation errors.
        profile_2_original_profile = Profile(
            depth_data=profile_2, pixel_size=pixel_size
        )
        profile_2_mod = apply_transform(profile_2_original_profile, transforms)

        # Compute correlation at this scale level (on filtered profiles)
        # MATLAB: current_profile2_lo = TranslateScalePointset(current_profile2_lo, [translation scaling])
        profile_2_filtered_profile = Profile(
            depth_data=profile_2_filtered, pixel_size=pixel_size
        )
        profile_2_filtered_transformed = apply_transform(
            profile_2_filtered_profile, transform
        )
        correlation_filtered = compute_cross_correlation(
            profile_1_filtered, profile_2_filtered_transformed
        )

        # Compute correlation on original (unfiltered) profiles after removing zeros
        # MATLAB: profiles2_scale_trans = TranslateScalePointset(mean_2, [translation scaling])
        # mean_2 here is the profile_2_mod from BEFORE this iteration (already has previous transforms)
        # But we need the version with current transform applied too.
        # Use profile_2_mod which now has ALL transforms applied.
        profile_1_no_zeros, profile_2_no_zeros, _ = _remove_boundary_zeros(
            profile_1, profile_2_mod
        )
        correlation_original = compute_cross_correlation(
            profile_1_no_zeros, profile_2_no_zeros
        )

        correlation_history.append((correlation_filtered, correlation_original))

    # Handle case where no scales were processed
    if len(transforms) == 0:
        transforms = [TransformParameters(translation=0.0, scaling=1.0)]
        correlation_history = [(np.nan, np.nan)]

    # Apply all transforms to get final aligned profile
    profile_2_profile = Profile(depth_data=profile_2, pixel_size=pixel_size)
    profile_2_aligned = apply_transform(profile_2_profile, transforms)

    # Optionally remove boundary zeros
    if params.remove_boundary_zeros:
        profile_1_aligned, profile_2_aligned, _ = _remove_boundary_zeros(
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

    :param reference: Reference profile (longer one).
    :param partial: Partial profile (shorter one) to align to reference.
    :param params: Alignment parameters. If None, default parameters are used.
    :param candidate_positions: Optional list of starting positions to try.
        If None, positions are determined by candidate search.
    :returns: Tuple of (AlignmentResult, best_start_position) where
        best_start_position is the index in the reference where the partial
        profile best aligns.
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

    # Expand candidates with neighboring positions (±2) for finer resolution.
    # The coarse candidate search may place region boundaries such that the
    # optimal position is narrowly missed.
    max_pos = len(ref_data) - partial_length // 2
    expanded = set(positions)
    for p in positions:
        for delta in range(-2, 3):
            nb = p + delta
            if 0 <= nb < max_pos:
                expanded.add(nb)
    positions = sorted(expanded)

    # Try each candidate position
    best_score = -np.inf
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

            # Check that alignment produced a meaningful overlap.
            # When the optimizer finds a spurious solution with an extreme shift,
            # only a few samples survive boundary zero removal. The correlation
            # on so few samples is unreliable. Require at least 50% overlap.
            aligned_length = len(result.reference_aligned)
            min_overlap = partial_length // 2
            if aligned_length < min_overlap:
                continue

            # Score candidates by correlation weighted by overlap ratio.
            # This prevents candidates with marginally higher correlation but
            # significantly less overlap from winning over better-aligned ones.
            overlap_ratio = aligned_length / partial_length
            score = result.final_correlation * overlap_ratio

            if score > best_score:
                best_score = score
                best_result = result
                best_start = candidate_start
        except ValueError:
            # Length mismatch - skip this candidate
            continue

    # If no valid result, create a default one
    if best_result is None:
        # Use position 0 and crop both to partial_length so that the
        # downstream code always receives equal-length profiles.
        ref_fallback = ref_data[:partial_length]
        best_result = AlignmentResult(
            transforms=(TransformParameters(translation=0.0, scaling=1.0),),
            correlation_history=np.array([[np.nan, np.nan]]),
            final_correlation=np.nan,
            reference_aligned=ref_fallback,
            compared_aligned=partial_data,
            total_translation=0.0,
            total_scaling=1.0,
        )

    return best_result, best_start
