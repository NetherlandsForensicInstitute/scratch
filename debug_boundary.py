"""Debug: check boundary clipping effect on objective function."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.alignment import (
    _alignment_objective,
    _apply_lowpass_filter_1d,
)

base = "packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel().astype(np.float64)
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel().astype(np.float64)
pixel_size = 3.5e-6

p1 = ref_data[:460]
p2 = comp_data.copy()
p1_lp = _apply_lowpass_filter_1d(p1, 500e-6, pixel_size)
p2_lp = _apply_lowpass_filter_1d(p2, 500e-6, pixel_size)
p1_sub = p1_lp[::15]
p2_sub = p2_lp[::15]

print(f"Profile lengths: {len(p1_sub)}, {len(p2_sub)}")

# Test the boundary clipping effect
for t in [0.0, -1e-15, -1e-14, -7.1e-15, -1e-10, -1e-5, -0.001, -0.01, -0.1, -1.0]:
    obj = _alignment_objective(np.array([t, 0.0]), p1_sub, p2_sub)
    print(f"  t={t:+15.2e}: obj={obj:.10f} (corr={-obj:.10f})")

print("\nPositive translations:")
for t in [0.0, 1e-15, 1e-14, 7.1e-15, 1e-10, 1e-5, 0.001, 0.01, 0.1, 1.0]:
    obj = _alignment_objective(np.array([t, 0.0]), p1_sub, p2_sub)
    print(f"  t={t:+15.2e}: obj={obj:.10f} (corr={-obj:.10f})")

# Check what happens with the transformed profile at these small shifts
from conversion.profile_correlator.data_types import Profile, TransformParameters
from conversion.profile_correlator.transforms import apply_transform

print("\nProfile values at boundaries:")
print(f"  p2_sub[0]  = {p2_sub[0]:.10f}")
print(f"  p2_sub[-1] = {p2_sub[-1]:.10f}")

for t in [0.0, -7.1e-15, 7.1e-15]:
    transform = TransformParameters(translation=t, scaling=1.0)
    prof = Profile(depth_data=p2_sub, pixel_size=1.0)
    p2_trans = apply_transform(prof, transform)
    print(f"\n  t={t:+.2e}: first 3 = {p2_trans[:3]}, last 3 = {p2_trans[-3:]}")
    # Count zeros
    n_zeros = np.sum(p2_trans == 0)
    print(f"    zeros: {n_zeros}, first_val={p2_trans[0]:.10f}, last_val={p2_trans[-1]:.10f}")
