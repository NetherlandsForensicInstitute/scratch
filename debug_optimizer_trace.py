"""Trace optimizer iterations for each scale pass."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.alignment import (
    _alignment_objective,
    _apply_lowpass_filter_1d,
    _fminsearchbnd_transform_to_bounded,
    _fminsearchbnd_transform_to_unconstrained,
)


def _matlab_fminsearch_traced(fun, x0, tol_x=1e-4, tol_fun=1e-4, max_iter=400, max_fun_evals=400, args=()):
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    n = len(x0)
    rho = 1.0
    chi = 2.0
    psi = 0.5
    sigma = 0.5
    usual_delta = 0.05
    zero_term_delta = 0.00025

    v = np.zeros((n + 1, n), dtype=np.float64)
    v[0] = x0.copy()
    for j in range(n):
        y = x0.copy()
        if y[j] != 0:
            y[j] = (1 + usual_delta) * y[j]
        else:
            y[j] = zero_term_delta
        v[j + 1] = y

    fv = np.zeros(n + 1, dtype=np.float64)
    for i in range(n + 1):
        fv[i] = fun(v[i], *args)
    func_evals = n + 1

    sort_idx = np.argsort(fv, kind="stable")
    fv = fv[sort_idx]
    v = v[sort_idx]

    iterations = 0
    exit_reason = "converged"

    while True:
        fv_diff = np.max(np.abs(fv[0] - fv[1:]))
        v_diff = np.max(np.abs(v[1:] - v[0]))

        if fv_diff <= max(tol_fun, 10 * np.finfo(np.float64).eps * abs(fv[0])) and v_diff <= max(
            tol_x, 10 * np.finfo(np.float64).eps * np.max(np.abs(v[0]))
        ):
            break
        if iterations >= max_iter:
            exit_reason = f"MAX_ITER({max_iter})"
            break
        if func_evals >= max_fun_evals:
            exit_reason = f"MAX_FEVALS({max_fun_evals})"
            break

        iterations += 1
        xbar = np.mean(v[:n], axis=0)
        xr = (1 + rho) * xbar - rho * v[n]
        fxr = fun(xr, *args)
        func_evals += 1

        if fxr < fv[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * v[n]
            fxe = fun(xe, *args)
            func_evals += 1
            if fxe < fxr:
                v[n] = xe
                fv[n] = fxe
            else:
                v[n] = xr
                fv[n] = fxr
        elif fxr < fv[n - 1]:
            v[n] = xr
            fv[n] = fxr
        elif fxr < fv[n]:
            xc = (1 + psi * rho) * xbar - psi * rho * v[n]
            fxc = fun(xc, *args)
            func_evals += 1
            if fxc <= fxr:
                v[n] = xc
                fv[n] = fxc
            else:
                for j in range(1, n + 1):
                    v[j] = v[0] + sigma * (v[j] - v[0])
                    fv[j] = fun(v[j], *args)
                func_evals += n
        else:
            xcc = (1 - psi) * xbar + psi * v[n]
            fxcc = fun(xcc, *args)
            func_evals += 1
            if fxcc < fv[n]:
                v[n] = xcc
                fv[n] = fxcc
            else:
                for j in range(1, n + 1):
                    v[j] = v[0] + sigma * (v[j] - v[0])
                    fv[j] = fun(v[j], *args)
                func_evals += n

        sort_idx = np.argsort(fv, kind="stable")
        fv = fv[sort_idx]
        v = v[sort_idx]

    return v[0], iterations, func_evals, fv[0], exit_reason


def fminsearchbnd_traced(fun, x0, lb, ub, tol_x=1e-6, tol_fun=1e-6, max_iter=400, max_fun_evals=400, args=()):
    x0u = _fminsearchbnd_transform_to_unconstrained(x0, lb, ub)

    def wrapped_fun(xu, *extra_args):
        x_bounded = _fminsearchbnd_transform_to_bounded(xu, lb, ub)
        return fun(x_bounded, *extra_args)

    xu_opt, iters, fevals, fval, exit_reason = _matlab_fminsearch_traced(
        wrapped_fun,
        x0u,
        tol_x=tol_x,
        tol_fun=tol_fun,
        max_iter=max_iter,
        max_fun_evals=max_fun_evals,
        args=args,
    )
    x_opt = _fminsearchbnd_transform_to_bounded(xu_opt, lb, ub)
    return x_opt, iters, fevals, fval, exit_reason


from conversion.profile_correlator.data_types import Profile, TransformParameters
from conversion.profile_correlator.transforms import apply_transform

for test_name in ["edge_over_threshold", "partial_with_nans"]:
    base = f"packages/scratch-core/tests/resources/profile_correlator/{test_name}"
    ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
    comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()
    pixel_size = 3.5e-6

    profile_1 = ref_data[: len(comp_data)].copy()
    profile_2 = comp_data.copy()

    transforms = []
    profile_2_mod = profile_2.copy()
    translation_total = 0.0
    scaling_total = 1.0
    current_scaling = 1.0
    resolution_limit = max(5e-6, 2 * pixel_size)

    scale_passes = [500e-6, 250e-6, 100e-6, 50e-6, 25e-6, 10e-6, 5e-6]
    _ftol = 1e-9

    print(f"\n{'=' * 70}")
    print(f"=== {test_name} (candidate 0, ref[0:{len(comp_data)}]) ===")
    print(f"{'=' * 70}")

    for cutoff in scale_passes:
        if cutoff < resolution_limit * (1 - _ftol):
            continue
        if cutoff > 1e-3 * (1 + _ftol):
            continue
        if cutoff < 5e-6 * (1 - _ftol):
            continue

        cutoff_um = cutoff * 1e6
        cutoff_pixels = cutoff / pixel_size
        subsample_factor = max(1, int(np.ceil(cutoff_pixels / 2 / 5)))

        p1_filt = _apply_lowpass_filter_1d(profile_1, cutoff, pixel_size, cut_borders=False)
        p2_filt = _apply_lowpass_filter_1d(profile_2_mod, cutoff, pixel_size, cut_borders=False)
        p1_sub = p1_filt[::subsample_factor]
        p2_sub = p2_filt[::subsample_factor]

        max_scaling_adj = 0.05 * (1 - (scaling_total - 1))
        min_scaling_adj = 0.05 * (1 + (scaling_total - 1))

        min_trans_adj = cutoff_um
        max_trans_adj = cutoff_um

        trans_lb = -int(round(min_trans_adj / subsample_factor))
        trans_ub = int(round(max_trans_adj / subsample_factor))
        scale_lb = ((1 - min_scaling_adj) / current_scaling - 1) * 10000
        scale_ub = ((1 + max_scaling_adj) / current_scaling - 1) * 10000

        lb = np.array([trans_lb, scale_lb], dtype=np.float64)
        ub = np.array([trans_ub, scale_ub], dtype=np.float64)
        x0 = np.array([0.0, 0.0])

        x_opt, iters, fevals, fval, exit_reason = fminsearchbnd_traced(
            _alignment_objective,
            x0,
            lb,
            ub,
            tol_x=1e-6,
            tol_fun=1e-6,
            max_iter=400,
            max_fun_evals=400,
            args=(p1_sub, p2_sub),
        )

        translation = x_opt[0] * subsample_factor
        scaling = x_opt[1] / 10000 + 1

        transform = TransformParameters(translation=translation, scaling=scaling)
        transforms.append(transform)
        current_scaling = scaling
        translation_total += translation
        scaling_total *= scaling

        p2_profile = Profile(depth_data=profile_2, pixel_size=pixel_size)
        profile_2_mod = apply_transform(p2_profile, transforms)

        print(
            f"  {cutoff_um:6.0f}um: trans={translation:9.4f} scale={scaling:.8f} "
            f"iters={iters:3d} fevals={fevals:3d} obj={fval:.8f} exit={exit_reason} "
            f"n_sub={len(p1_sub)} bounds=[{trans_lb},{trans_ub}]"
        )

    print(f"\n  TOTAL: trans={translation_total:.4f} samples ({translation_total * pixel_size * 1e6:.2f} um)")
    print(f"  TOTAL: scale={scaling_total:.8f}")
