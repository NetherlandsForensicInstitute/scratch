"""Evaluate the objective landscape to understand optimizer convergence."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.alignment import (
    _alignment_objective,
    _apply_lowpass_filter_1d,
)

# Load edge_over_threshold data
base = "packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()
pixel_size = 3.5e-6

profile_1 = ref_data[:460].copy()
profile_2 = comp_data.copy()

# Check all scale passes
scale_passes_um = [500, 250, 100, 50, 25, 10, 5]
cutoff_lo = 5e-6
resolution_limit = max(cutoff_lo, 2 * pixel_size)

for scale_um in scale_passes_um:
    cutoff = scale_um * 1e-6
    cutoff_pixels = cutoff / pixel_size
    subsample_factor = max(1, int(np.ceil(cutoff_pixels / 2 / 5)))

    p1_filt = _apply_lowpass_filter_1d(profile_1, cutoff, pixel_size, cut_borders=False)
    p2_filt = _apply_lowpass_filter_1d(profile_2, cutoff, pixel_size, cut_borders=False)
    p1_sub = p1_filt[::subsample_factor]
    p2_sub = p2_filt[::subsample_factor]

    # Compute bounds (redetermine_max_trans=True)
    trans_bound = int(round(scale_um / subsample_factor))

    # Scan translation at scaling=0
    best_t = 0
    best_obj = 999
    print(
        f"\n=== Scale {scale_um}um (sub={subsample_factor}, n={len(p1_sub)}, trans_bound=[-{trans_bound},{trans_bound}]) ==="
    )
    for t in range(-min(trans_bound, 20), min(trans_bound, 20) + 1):
        obj = _alignment_objective(np.array([float(t), 0.0]), p1_sub, p2_sub)
        marker = " <-- " if abs(t) <= 1 else ""
        if t in [-3, -2, -1, 0, 1, 2, 3] or (abs(obj) > 0.9 and abs(t) < 5):
            print(f"  t={t:6.1f}: obj={obj:.10f}{marker}")
        if obj < best_obj:
            best_obj = obj
            best_t = t
    print(f"  Best: t={best_t}, obj={best_obj:.10f}")

# Now check: what if we DON'T have the sin-transform artifact?
# i.e., what if interp1 correctly handles boundary values?
print("\n\n=== EFFECT OF BOUNDARY HANDLING ===")
cutoff = 500e-6
subsample_factor = 15
p1_filt = _apply_lowpass_filter_1d(profile_1, cutoff, pixel_size, cut_borders=False)
p2_filt = _apply_lowpass_filter_1d(profile_2, cutoff, pixel_size, cut_borders=False)
p1_sub = p1_filt[::subsample_factor]
p2_sub = p2_filt[::subsample_factor]

from conversion.profile_correlator.data_types import Profile, TransformParameters
from conversion.profile_correlator.similarity import compute_cross_correlation
from conversion.profile_correlator.transforms import apply_transform

# Direct correlation (no transform)
corr_direct = compute_cross_correlation(p1_sub, p2_sub)
print(f"Direct correlation (no transform): {corr_direct:.15f}")

# With identity transform (translation=0, scaling=1)
transform_id = TransformParameters(translation=0.0, scaling=1.0)
p2_prof = Profile(depth_data=p2_sub, pixel_size=1.0)
p2_id = apply_transform(p2_prof, transform_id)
corr_id = compute_cross_correlation(p1_sub, p2_id)
print(f"Identity transform correlation:    {corr_id:.15f}")
print(f"Identity transformed == original:  {np.array_equal(p2_sub, p2_id)}")
print(f"Max diff: {np.max(np.abs(p2_sub - p2_id)):.2e}")
print(f"Last 3 original: {p2_sub[-3:]}")
print(f"Last 3 transformed: {p2_id[-3:]}")

# Check: is zero at the end the issue?
p2_fixed = p2_id.copy()
p2_fixed[-1] = p2_sub[-1]  # restore the zeroed last sample
corr_fixed = compute_cross_correlation(p1_sub, p2_fixed)
print(f"\nFixed last sample correlation:     {corr_fixed:.15f}")
print(f"Original direct correlation:       {corr_direct:.15f}")
print(f"Match? {np.isclose(corr_fixed, corr_direct)}")
