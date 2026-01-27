"""Trace the exact optimizer initial simplex for edge_over_threshold at 500um."""
import sys
sys.path.insert(0, 'packages/scratch-core/src')

import numpy as np
from conversion.profile_correlator.alignment import (
    _apply_lowpass_filter_1d,
    _alignment_objective,
    _fminsearchbnd_transform_to_unconstrained,
    _fminsearchbnd_transform_to_bounded,
)
from conversion.profile_correlator.transforms import apply_transform
from conversion.profile_correlator.data_types import Profile, TransformParameters
from conversion.profile_correlator.similarity import compute_cross_correlation

# Load edge_over_threshold data
base = 'packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold'
ref_data = np.load(f'{base}/input_profile_ref.npy').ravel()
comp_data = np.load(f'{base}/input_profile_comp.npy').ravel()
pixel_size = 3.5e-6

# Setup: candidate_start=0, partial profile
profile_1 = ref_data[:460].copy()  # ref segment matching comp length
profile_2 = comp_data.copy()

# 500um scale pass parameters
cutoff = 500e-6  # meters
cutoff_pixels = cutoff / pixel_size
subsample_factor = max(1, int(np.ceil(cutoff_pixels / 2 / 5)))

# Filter
p1_filt = _apply_lowpass_filter_1d(profile_1, cutoff, pixel_size, cut_borders=False)
p2_filt = _apply_lowpass_filter_1d(profile_2, cutoff, pixel_size, cut_borders=False)

# Subsample
p1_sub = p1_filt[::subsample_factor]
p2_sub = p2_filt[::subsample_factor]

print(f"Profile lengths: ref={len(profile_1)}, comp={len(profile_2)}")
print(f"cutoff_pixels={cutoff_pixels:.4f}, subsample_factor={subsample_factor}")
print(f"Subsampled lengths: p1={len(p1_sub)}, p2={len(p2_sub)}")

# Bounds (redetermine_max_trans=True)
cutoff_um = cutoff * 1e6
trans_lb = -int(round(cutoff_um / subsample_factor))
trans_ub = int(round(cutoff_um / subsample_factor))
max_scaling = 0.05
scale_lb = ((1 - max_scaling) / 1.0 - 1) * 10000
scale_ub = ((1 + max_scaling) / 1.0 - 1) * 10000

lb = np.array([trans_lb, scale_lb], dtype=np.float64)
ub = np.array([trans_ub, scale_ub], dtype=np.float64)

print(f"\nBounds: lb={lb}, ub={ub}")

# Test the objective at [0, 0] directly
obj_direct = _alignment_objective(np.array([0.0, 0.0]), p1_sub, p2_sub)
print(f"\nObjective at direct [0,0]: {obj_direct:.15f}")

# Now trace the sin transformation
x0 = np.array([0.0, 0.0])
x0u = _fminsearchbnd_transform_to_unconstrained(x0, lb, ub)
print(f"\nUnconstrained x0: {x0u}")
print(f"Expected: [2*pi, 2*pi] = [{2*np.pi}, {2*np.pi}]")

# Transform back to bounded
x0_back = _fminsearchbnd_transform_to_bounded(x0u, lb, ub)
print(f"Bounded x0 (back-transformed): {x0_back}")
print(f"sin(2*pi) = {np.sin(2*np.pi):.20e}")

# Objective at the sin-transformed initial point
obj_sin = _alignment_objective(x0_back, p1_sub, p2_sub)
print(f"Objective at sin-transformed [0,0]: {obj_sin:.15f}")
print(f"Difference from direct: {obj_sin - obj_direct:.2e}")

# Now build the initial simplex as fminsearch would
n = 2
usual_delta = 0.05
zero_term_delta = 0.00025

v = np.zeros((n + 1, n), dtype=np.float64)
v[0] = x0u.copy()
for j in range(n):
    y = x0u.copy()
    if y[j] != 0:
        y[j] = (1 + usual_delta) * y[j]
    else:
        y[j] = zero_term_delta
    v[j + 1] = y

print(f"\n=== INITIAL SIMPLEX (unconstrained) ===")
for i in range(3):
    print(f"v[{i}] = {v[i]}")

# Transform each vertex to bounded space and evaluate
print(f"\n=== INITIAL SIMPLEX EVALUATIONS ===")
fv = np.zeros(3)
for i in range(3):
    x_bounded = _fminsearchbnd_transform_to_bounded(v[i], lb, ub)
    fv[i] = _alignment_objective(x_bounded, p1_sub, p2_sub)
    print(f"v[{i}]: bounded={x_bounded}, obj={fv[i]:.15f}")

# Sort
sort_idx = np.argsort(fv, kind="stable")
print(f"\nSort order: {sort_idx}")
print(f"Best vertex: {sort_idx[0]} with obj={fv[sort_idx[0]]:.15f}")

# Check what the transformed profiles look like at v[0]
x_v0 = _fminsearchbnd_transform_to_bounded(v[0], lb, ub)
transform_v0 = TransformParameters(translation=x_v0[0], scaling=x_v0[1]/10000 + 1)
p2_prof = Profile(depth_data=p2_sub, pixel_size=1.0)
p2_transformed = apply_transform(p2_prof, transform_v0)

print(f"\n=== v[0] transform details ===")
print(f"Translation: {x_v0[0]:.20e}")
print(f"Scaling: {x_v0[1]/10000 + 1:.20e}")
print(f"p2_sub[-3:]: {p2_sub[-3:]}")
print(f"p2_transformed[-3:]: {p2_transformed[-3:]}")
print(f"p2_transformed[0:3]: {p2_transformed[0:3]}")
print(f"Zeros in p2_transformed: {np.sum(p2_transformed == 0)}")
print(f"Correlation: {compute_cross_correlation(p1_sub, p2_transformed):.15f}")

# Now run with FULL optimizer tracing
print(f"\n=== RUNNING FULL OPTIMIZER ===")
from conversion.profile_correlator.alignment import _fminsearchbnd
x_opt = _fminsearchbnd(
    _alignment_objective, x0, lb, ub,
    tol_x=1e-6, tol_fun=1e-6, max_iter=400, max_fun_evals=400,
    args=(p1_sub, p2_sub),
)
print(f"\nOptimizer result: {x_opt}")
print(f"Translation (subsampled): {x_opt[0]:.10f}")
print(f"Scaling: {x_opt[1]/10000 + 1:.10f}")
print(f"Translation (full): {x_opt[0] * subsample_factor:.6f}")
