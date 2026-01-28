"""Debug script to trace alignment step by step for edge_over_threshold."""

import sys

sys.path.insert(0, "/Users/laurensweijs/scratch/packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.alignment import (
    _alignment_objective,
    _apply_lowpass_filter_1d,
    _fminsearchbnd,
    _remove_boundary_zeros,
    align_profiles_multiscale,
)
from conversion.profile_correlator.candidate_search import find_match_candidates
from conversion.profile_correlator.data_types import AlignmentParameters, Profile, TransformParameters
from conversion.profile_correlator.similarity import compute_cross_correlation
from conversion.profile_correlator.transforms import apply_transform

# Load data
base = "/Users/laurensweijs/scratch/packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()

pixel_size = 3.5e-6
profile_ref = Profile(depth_data=ref_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)
profile_comp = Profile(depth_data=comp_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)

# Candidate search
cand_pos, shape_scales, comp_scale = find_match_candidates(
    profile_ref, profile_comp, AlignmentParameters(cutoff_hi=1e-3, cutoff_lo=5e-6)
)
print(f"Candidates: {cand_pos}, comp_scale: {comp_scale * 1e6:.0f} um")

# Simulate partial alignment for candidate 0
candidate_start = int(cand_pos[0])
partial_length = len(comp_data)
ref_segment = ref_data[candidate_start : candidate_start + partial_length]

# Create profiles for alignment
ref_prof = Profile(depth_data=ref_segment, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)
comp_prof = Profile(depth_data=comp_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)

# Scale passes (adjusted to <= comp_scale)
all_passes = (1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6)
adjusted_passes = tuple(s for s in all_passes if s <= comp_scale)
print(f"Adjusted passes (um): {[p * 1e6 for p in adjusted_passes]}")

# Resolution limit
resolution_limit = max(5e-6, 2 * pixel_size)
cutoff_hi = 1e-3
cutoff_lo = 5e-6
print(f"Resolution limit: {resolution_limit * 1e6} um")

# Run alignment manually, tracking each step
profile_1 = ref_segment.copy()
profile_2 = comp_data.copy()

transforms = []
translation_total = 0.0
scaling_total = 1.0
current_scaling = 1.0
profile_2_mod = profile_2.copy()
_ftol = 1e-9

print(f"\n{'=' * 80}")
print("ALIGNMENT STEP-BY-STEP")
print(f"Profile lengths: ref={len(profile_1)}, comp={len(profile_2)}")
print(f"{'=' * 80}")

for cutoff in adjusted_passes:
    cutoff_um = cutoff * 1e6

    # Check if should process
    if cutoff < resolution_limit * (1 - _ftol):
        print(f"\nScale {cutoff_um:.0f} um: SKIPPED (below resolution limit)")
        continue
    if cutoff > cutoff_hi * (1 + _ftol):
        print(f"\nScale {cutoff_um:.0f} um: SKIPPED (above cutoff_hi)")
        continue
    if cutoff < cutoff_lo * (1 - _ftol):
        print(f"\nScale {cutoff_um:.0f} um: SKIPPED (below cutoff_lo)")
        continue

    print(f"\n--- Scale {cutoff_um:.0f} um ---")

    # Lowpass filter
    p1_filtered = _apply_lowpass_filter_1d(profile_1, cutoff, pixel_size)
    p2_filtered = _apply_lowpass_filter_1d(profile_2_mod, cutoff, pixel_size)

    # Subsample
    cutoff_samples = cutoff / pixel_size
    subsample_factor = max(1, int(np.ceil(cutoff_samples / 2 / 5)))
    p1_sub = p1_filtered[::subsample_factor]
    p2_sub = p2_filtered[::subsample_factor]

    # Bounds (redetermine_max_trans = True since max_translation = 10e6 um)
    trans_lb = -int(round(cutoff_um / subsample_factor))
    trans_ub = int(round(cutoff_um / subsample_factor))

    max_scaling = 0.05
    max_scaling_adj = max_scaling * (1 - (scaling_total - 1))
    min_scaling_adj = max_scaling * (1 + (scaling_total - 1))
    scale_lb = ((1 - min_scaling_adj) / current_scaling - 1) * 10000
    scale_ub = ((1 + max_scaling_adj) / current_scaling - 1) * 10000

    lb = np.array([trans_lb, scale_lb], dtype=np.float64)
    ub = np.array([trans_ub, scale_ub], dtype=np.float64)
    x0 = np.array([0.0, 0.0])

    print(f"  Subsample factor: {subsample_factor}")
    print(f"  Sub lengths: p1={len(p1_sub)}, p2={len(p2_sub)}")
    print(f"  Bounds: trans=[{trans_lb}, {trans_ub}], scale=[{scale_lb:.2f}, {scale_ub:.2f}]")

    # Evaluate objective at a few key points
    obj_0 = _alignment_objective(np.array([0.0, 0.0]), p1_sub, p2_sub)
    print(f"  Objective at [0,0]: {obj_0:.6f} (corr={-obj_0:.6f})")

    # Optimize
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

    print(f"  Optimized: trans_sub={translation_sub:.4f}, trans_orig={translation:.4f}, scale={scaling:.6f}")

    obj_opt = _alignment_objective(x_opt, p1_sub, p2_sub)
    print(f"  Objective at optimum: {obj_opt:.6f} (corr={-obj_opt:.6f})")

    # Update transforms
    transform = TransformParameters(translation=translation, scaling=scaling)
    transforms.append(transform)

    current_scaling = scaling
    translation_total += translation
    scaling_total *= scaling

    # Recompute profile_2_mod from original
    p2_orig_prof = Profile(depth_data=profile_2, pixel_size=pixel_size)
    profile_2_mod = apply_transform(p2_orig_prof, transforms)

    # Correlation after zero removal
    p1_nz, p2_nz, _ = _remove_boundary_zeros(profile_1, profile_2_mod)
    corr_nz = compute_cross_correlation(p1_nz, p2_nz)
    print(f"  After transform: total_trans={translation_total:.4f}, total_scale={scaling_total:.6f}")
    print(f"  Non-zero overlap: {len(p1_nz)} samples, corr={corr_nz:.6f}")

print(f"\n{'=' * 80}")
print("FINAL RESULT")
print(f"  Total translation: {translation_total:.4f} samples = {translation_total * pixel_size * 1e6:.2f} um")
print(f"  Total scaling: {scaling_total:.6f}")
print(f"  Non-zero overlap: {len(p1_nz)} samples")
print(f"  pOverlap = {len(p1_nz) * pixel_size / (partial_length * pixel_size):.6f}")
print("\n  MATLAB expected: dPos=-123.92 um, pOverlap=0.924, simVal=0.986")

# Now also run the full alignment to compare
print(f"\n{'=' * 80}")
print("RUNNING FULL align_profiles_multiscale")
params = AlignmentParameters(scale_passes=adjusted_passes, cutoff_hi=1e-3, cutoff_lo=5e-6)
result = align_profiles_multiscale(ref_prof, comp_prof, params)
print(
    f"  Total translation: {result.total_translation:.4f} samples = {result.total_translation * pixel_size * 1e6:.2f} um"
)
print(f"  Total scaling: {result.total_scaling:.6f}")
print(f"  Final correlation: {result.final_correlation:.6f}")
print(f"  Aligned lengths: ref={len(result.reference_aligned)}, comp={len(result.compared_aligned)}")
print(f"  pOverlap = {len(result.reference_aligned) * pixel_size / (partial_length * pixel_size):.6f}")
