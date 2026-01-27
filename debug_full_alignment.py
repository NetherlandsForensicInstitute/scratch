"""Trace the full multi-scale alignment for edge_over_threshold."""
import sys
sys.path.insert(0, 'packages/scratch-core/src')

import numpy as np
from conversion.profile_correlator.alignment import (
    _apply_lowpass_filter_1d,
    _alignment_objective,
    _fminsearchbnd,
    _remove_boundary_zeros,
)
from conversion.profile_correlator.data_types import Profile, AlignmentParameters, TransformParameters
from conversion.profile_correlator.transforms import apply_transform, compute_cumulative_transform
from conversion.profile_correlator.candidate_search import find_match_candidates
from conversion.profile_correlator.similarity import compute_cross_correlation

# Load edge_over_threshold data
base = 'packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold'
ref_data = np.load(f'{base}/input_profile_ref.npy').ravel()
comp_data = np.load(f'{base}/input_profile_comp.npy').ravel()
pixel_size = 3.5e-6

ref_profile = Profile(depth_data=ref_data, pixel_size=pixel_size)
comp_profile = Profile(depth_data=comp_data, pixel_size=pixel_size)
params = AlignmentParameters()

# Find candidates
positions, shape_scales, comp_scale = find_match_candidates(ref_profile, comp_profile, params)
adjusted_passes = tuple(s for s in params.scale_passes if s <= comp_scale)

print(f"Candidates: {positions}")
print(f"Comp scale: {comp_scale*1e6:.0f}um")
print(f"Passes: {[f'{s*1e6:.0f}um' for s in adjusted_passes]}")

# Run alignment for candidate 0
candidate_start = positions[0]
ref_segment = ref_data[candidate_start:candidate_start + len(comp_data)]
profile_1 = ref_segment.copy()
profile_2 = comp_data.copy()

resolution_limit = max(params.cutoff_lo, 2 * pixel_size)

translation_total = 0.0
scaling_total = 1.0
current_scaling = 1.0
transforms = []
profile_2_mod = profile_2.copy()

_ftol = 1e-9

for i, cutoff in enumerate(adjusted_passes):
    cutoff_um = cutoff * 1e6
    if cutoff < resolution_limit * (1 - _ftol):
        continue
    if cutoff > params.cutoff_hi * (1 + _ftol):
        continue
    if cutoff < params.cutoff_lo * (1 - _ftol):
        continue

    # Apply lowpass filter
    p1_filt = _apply_lowpass_filter_1d(profile_1, cutoff, pixel_size, cut_borders=params.cut_borders_after_smoothing)
    p2_filt = _apply_lowpass_filter_1d(profile_2_mod, cutoff, pixel_size, cut_borders=params.cut_borders_after_smoothing)

    # Subsample
    cutoff_pixels = cutoff / pixel_size
    subsample_factor = max(1, int(np.ceil(cutoff_pixels / 2 / 5)))
    p1_sub = p1_filt[::subsample_factor]
    p2_sub = p2_filt[::subsample_factor]

    # Bounds
    max_trans_adj = params.max_translation / pixel_size - translation_total
    min_trans_adj = params.max_translation / pixel_size + translation_total
    max_scaling_adj = params.max_scaling * (1 - (scaling_total - 1))
    min_scaling_adj = params.max_scaling * (1 + (scaling_total - 1))

    # Redetermine max_trans (default max_translation=10.0m → 10000mm → redetermine)
    max_translation_mm = params.max_translation * 1000
    redetermine_max_trans = abs(max_translation_mm - 10000) < 1e-6
    if redetermine_max_trans:
        min_trans_adj = cutoff_um
        max_trans_adj = cutoff_um

    trans_lb = -int(round(min_trans_adj / subsample_factor))
    trans_ub = int(round(max_trans_adj / subsample_factor))
    scale_lb = ((1 - min_scaling_adj) / current_scaling - 1) * 10000
    scale_ub = ((1 + max_scaling_adj) / current_scaling - 1) * 10000

    lb = np.array([trans_lb, scale_lb], dtype=np.float64)
    ub = np.array([trans_ub, scale_ub], dtype=np.float64)

    x0 = np.array([0.0, 0.0])

    # Run optimizer
    x_opt = _fminsearchbnd(
        _alignment_objective, x0, lb, ub,
        tol_x=1e-6, tol_fun=1e-6, max_iter=400, max_fun_evals=400,
        args=(p1_sub, p2_sub),
    )

    translation_subsampled = x_opt[0]
    scaling_encoded = x_opt[1]
    translation = translation_subsampled * subsample_factor
    scaling = scaling_encoded / 10000.0 + 1.0

    transform = TransformParameters(translation=translation, scaling=scaling)
    transforms.append(transform)

    current_scaling = scaling
    translation_total += translation
    scaling_total *= scaling

    # Recompute profile_2_mod from original
    p2_profile = Profile(depth_data=profile_2, pixel_size=pixel_size)
    profile_2_mod = apply_transform(p2_profile, transforms)

    # Compute correlations
    p2_filt_profile = Profile(depth_data=p2_filt, pixel_size=pixel_size)
    p2_filt_transformed = apply_transform(p2_filt_profile, transform)
    corr_filtered = compute_cross_correlation(p1_filt, p2_filt_transformed)

    p1_nz, p2_nz, _ = _remove_boundary_zeros(profile_1, profile_2_mod)
    corr_original = compute_cross_correlation(p1_nz, p2_nz)

    overlap_len = len(p1_nz)
    overlap_ratio = overlap_len / len(profile_2) if len(profile_2) > 0 else 0

    print(f"\nScale {cutoff_um:.0f}um (pass {i+1}/{len(adjusted_passes)}):")
    print(f"  subsample={subsample_factor}, sub_len={len(p1_sub)}, bounds=[{trans_lb},{trans_ub}]")
    print(f"  x_opt_sub=[{translation_subsampled:.6f}, {scaling_encoded:.6f}]")
    print(f"  translation={translation:.4f} samples, scaling={scaling:.8f}")
    print(f"  total: trans={translation_total:.4f}, scale={scaling_total:.8f}")
    print(f"  corr_filtered={corr_filtered:.6f}, corr_original={corr_original:.6f}")
    print(f"  overlap_len={overlap_len}/{len(profile_2)}, overlap_ratio={overlap_ratio:.4f}")

# Final result
if params.remove_boundary_zeros:
    p1_aligned, p2_aligned, _ = _remove_boundary_zeros(profile_1, profile_2_mod)
else:
    p1_aligned = profile_1
    p2_aligned = profile_2_mod

final_corr = compute_cross_correlation(p1_aligned, p2_aligned)
pOverlap = len(p2_aligned) / len(profile_2)

print(f"\n{'='*60}")
print(f"FINAL RESULT:")
print(f"  ccf = {final_corr:.6f} (expected: 0.986387)")
print(f"  pOverlap = {pOverlap:.6f} (expected: 0.923913)")
print(f"  total_translation = {translation_total:.4f}")
print(f"  total_scaling = {scaling_total:.8f}")
print(f"  aligned lengths: ref={len(p1_aligned)}, comp={len(p2_aligned)}")
