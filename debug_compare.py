"""Debug script to compare Python's result vs MATLAB's expected result for edge_over_threshold."""

import sys

sys.path.insert(0, "/Users/laurensweijs/scratch/packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.alignment import _apply_lowpass_filter_1d
from conversion.profile_correlator.correlator import correlate_profiles
from conversion.profile_correlator.data_types import AlignmentParameters, Profile
from conversion.profile_correlator.similarity import compute_cross_correlation

base = "/Users/laurensweijs/scratch/packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()

pixel_size = 3.5e-6
params = AlignmentParameters(
    cutoff_hi=1e-3,
    cutoff_lo=5e-6,
    partial_mark_threshold=8.0,
    inclusion_threshold=0.5,
)

profile_ref = Profile(depth_data=ref_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)
profile_comp = Profile(depth_data=comp_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)

print(f"ref: {len(ref_data)}, comp: {len(comp_data)}")
print(f"length diff: {abs(len(ref_data) - len(comp_data)) / max(len(ref_data), len(comp_data)) * 100:.1f}%")

# Run Python correlator
result = correlate_profiles(profile_ref, profile_comp, params)
print("\n=== Python result ===")
print(f"dPos: {result.position_shift * 1e6:.2f} um")
print(f"dScale: {result.scale_factor:.6f}")
print(f"simVal: {result.similarity_value:.6f}")
print(f"pOverlap: {result.overlap_ratio:.6f}")
print(f"lOverlap: {result.overlap_length * 1e6:.2f} um")

print("\n=== MATLAB expected ===")
print("dPos: -123.92 um")
print("dScale: 1.001523")
print("simVal: 0.986387")
print("pOverlap: 0.923913")
print("lOverlap: 1487.50 um")

# Now compute what MATLAB's result would look like:
# MATLAB found a translation of -35.4 samples = -123.92 um
# Let's verify: at that shift, what is the correlation?
ref_seg = ref_data[:460].astype(np.float64)
comp_seg = comp_data.astype(np.float64).copy()

# Apply MATLAB's transform: shift by -35.4 samples, scale by 1.00152
from conversion.profile_correlator.alignment import _remove_boundary_zeros
from conversion.profile_correlator.data_types import TransformParameters
from conversion.profile_correlator.transforms import apply_transform

matlab_trans = -123.92e-6 / pixel_size  # -35.4 samples
matlab_scale = 1.0015232185573393
transform = TransformParameters(translation=matlab_trans, scaling=matlab_scale)
comp_prof = Profile(depth_data=comp_seg, pixel_size=pixel_size)
comp_transformed = apply_transform(comp_prof, transform)

# Remove boundary zeros
ref_nz, comp_nz, start = _remove_boundary_zeros(ref_seg, comp_transformed)
corr_matlab = compute_cross_correlation(ref_nz, comp_nz)
print("\n=== Simulating MATLAB's result ===")
print(f"After transform with t={matlab_trans:.4f}, s={matlab_scale:.6f}")
print(f"Overlap: {len(ref_nz)} samples")
print(f"pOverlap: {len(ref_nz) / 460:.6f}")
print(f"Correlation: {corr_matlab:.6f}")

# Compare with Python's t=0 result
corr_python = compute_cross_correlation(ref_seg, comp_seg)
print("\n=== At t=0, s=1.0 (Python finds) ===")
print(f"Correlation (full): {corr_python:.6f}")
print("pOverlap: 1.0")

# Check objective at each scale for both solutions
print("\n=== Objective function comparison at first scale (500 um) ===")
p1_lp = _apply_lowpass_filter_1d(ref_seg, 500e-6, pixel_size)
p2_lp = _apply_lowpass_filter_1d(comp_seg, 500e-6, pixel_size)

subsample = 15
p1_sub = p1_lp[::subsample]
p2_sub = p2_lp[::subsample]

# At t=0
from conversion.profile_correlator.alignment import _alignment_objective

obj_0 = _alignment_objective(np.array([0.0, 0.0]), p1_sub, p2_sub)
print(f"Objective at t=0: {obj_0:.6f} (corr={-obj_0:.6f})")

# At t=-35.4 samples = -2.36 subsampled
t_sub = matlab_trans / subsample
obj_matlab = _alignment_objective(np.array([t_sub, (matlab_scale - 1) * 10000]), p1_sub, p2_sub)
print(f"Objective at t={t_sub:.4f} (MATLAB): {obj_matlab:.6f} (corr={-obj_matlab:.6f})")

# Scan translation to find all local optima
print("\n=== Objective landscape scan at 500 um scale ===")
for t in np.arange(-33, 34, 1):
    obj = _alignment_objective(np.array([float(t), 0.0]), p1_sub, p2_sub)
    marker = " <-- MATLAB" if abs(t - t_sub) < 1 else (" <-- Python" if abs(t) < 0.5 else "")
    if abs(t) <= 5 or abs(t - t_sub) <= 3 or t % 5 == 0:
        print(f"  t={t:+6.1f}: corr={-obj:.6f}{marker}")
