"""Debug: trace objective landscape at each scale for edge_over_threshold."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.alignment import (
    _alignment_objective,
    _apply_lowpass_filter_1d,
    _fminsearchbnd,
    _remove_boundary_zeros,
)
from conversion.profile_correlator.data_types import Profile, TransformParameters
from conversion.profile_correlator.similarity import compute_cross_correlation
from conversion.profile_correlator.transforms import apply_transform

base = "packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel().astype(np.float64)
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel().astype(np.float64)
pixel_size = 3.5e-6

profile_1 = ref_data[:460].copy()  # ref segment for candidate 0
profile_2 = comp_data.copy()

transforms = []
translation_total = 0.0
scaling_total = 1.0
current_scaling = 1.0
profile_2_mod = profile_2.copy()

passes = [500e-6, 250e-6, 100e-6, 50e-6, 25e-6, 10e-6]
resolution_limit = 7e-6

for cutoff in passes:
    cutoff_um = cutoff * 1e6

    p1_filtered = _apply_lowpass_filter_1d(profile_1, cutoff, pixel_size)
    p2_filtered = _apply_lowpass_filter_1d(profile_2_mod, cutoff, pixel_size)

    cutoff_samples = cutoff / pixel_size
    subsample_factor = max(1, int(np.ceil(cutoff_samples / 2 / 5)))

    p1_sub = p1_filtered[::subsample_factor]
    p2_sub = p2_filtered[::subsample_factor]

    # Bounds
    trans_lb = -int(round(cutoff_um / subsample_factor))
    trans_ub = int(round(cutoff_um / subsample_factor))
    max_scaling_adj = 0.05 * (1 - (scaling_total - 1))
    min_scaling_adj = 0.05 * (1 + (scaling_total - 1))
    scale_lb = ((1 - min_scaling_adj) / current_scaling - 1) * 10000
    scale_ub = ((1 + max_scaling_adj) / current_scaling - 1) * 10000

    lb = np.array([trans_lb, scale_lb], dtype=np.float64)
    ub = np.array([trans_ub, scale_ub], dtype=np.float64)
    x0 = np.array([0.0, 0.0])

    # Landscape scan at this scale
    obj_0 = _alignment_objective(np.array([0.0, 0.0]), p1_sub, p2_sub)

    # Find minimum by brute force
    best_t = 0
    best_obj = obj_0
    for t in range(trans_lb, trans_ub + 1):
        obj = _alignment_objective(np.array([float(t), 0.0]), p1_sub, p2_sub)
        if obj < best_obj:
            best_obj = obj
            best_t = t

    # Run optimizer
    x_opt = _fminsearchbnd(
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

    translation_sub = x_opt[0]
    scaling_enc = x_opt[1]
    translation = translation_sub * subsample_factor
    scaling = scaling_enc / 10000.0 + 1.0

    print(
        f"Scale {cutoff_um:5.0f} um: sub={subsample_factor}, "
        f"len=[{len(p1_sub)},{len(p2_sub)}], "
        f"bounds=[{trans_lb},{trans_ub}]"
    )
    print(f"  Brute-force best: t={best_t}, corr={-best_obj:.6f}")
    print(f"  Objective at t=0: corr={-obj_0:.6f}")
    print(
        f"  Optimizer result: t_sub={translation_sub:.4f}, "
        f"trans={translation:.4f}, scale={scaling:.6f}, "
        f"corr={-_alignment_objective(x_opt, p1_sub, p2_sub):.6f}"
    )

    transform = TransformParameters(translation=translation, scaling=scaling)
    transforms.append(transform)
    current_scaling = scaling
    translation_total += translation
    scaling_total *= scaling

    profile_2_original_profile = Profile(depth_data=profile_2, pixel_size=pixel_size)
    profile_2_mod = apply_transform(profile_2_original_profile, transforms)

    p1_nz, p2_nz, _ = _remove_boundary_zeros(profile_1, profile_2_mod)
    print(
        f"  After transform: total_trans={translation_total:.4f} samples, "
        f"overlap={len(p1_nz)}, corr={compute_cross_correlation(p1_nz, p2_nz):.6f}"
    )
    print()

print(f"FINAL: total_trans={translation_total:.4f} samples = {translation_total * pixel_size * 1e6:.2f} um")
print(f"MATLAB expected: -123.92 um = {-123.92 / 3.5:.2f} samples")
