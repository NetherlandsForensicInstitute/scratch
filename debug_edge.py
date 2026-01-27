"""
Debug script for edge_over_threshold test case.

Investigates why Python finds near-zero translation (full overlap) while
MATLAB finds ~35 sample translation (partial overlap of 0.924).

Expected MATLAB results from metadata.json:
  - dPos = -123.92 um => translation ~ -123.92/3.5 ~ -35.4 samples
  - dScale = 1.0015232185573393
  - pOverlap = 0.9239130434782609
  - ccf = 0.9863872230983314
  - startPartProfile = 3.5 um => candidate position = 1 sample
"""

import sys
sys.path.insert(0, '/Users/laurensweijs/scratch/packages/scratch-core/src')

import numpy as np
from conversion.profile_correlator.data_types import Profile, AlignmentParameters, TransformParameters
from conversion.profile_correlator.candidate_search import find_match_candidates
from conversion.profile_correlator.alignment import _apply_lowpass_filter_1d, _alignment_objective
from conversion.profile_correlator.transforms import apply_transform
from conversion.profile_correlator.similarity import compute_cross_correlation

# ============================================================================
# 1. Load test data
# ============================================================================
data_dir = '/Users/laurensweijs/scratch/packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold'

ref_data = np.load(f'{data_dir}/input_profile_ref.npy')
comp_data = np.load(f'{data_dir}/input_profile_comp.npy')

print("=" * 70)
print("1. RAW DATA")
print("=" * 70)
print(f"ref_data shape:  {ref_data.shape}, dtype: {ref_data.dtype}")
print(f"comp_data shape: {comp_data.shape}, dtype: {comp_data.dtype}")
print(f"ref_data range:  [{np.nanmin(ref_data):.6e}, {np.nanmax(ref_data):.6e}]")
print(f"comp_data range: [{np.nanmin(comp_data):.6e}, {np.nanmax(comp_data):.6e}]")
print(f"ref NaN count:   {np.sum(np.isnan(ref_data))}")
print(f"comp NaN count:  {np.sum(np.isnan(comp_data))}")

# Handle multi-column data
if ref_data.ndim == 2:
    print(f"ref columns: {ref_data.shape[1]}, using mean across columns")
    ref_mean = np.nanmean(ref_data, axis=1)
else:
    ref_mean = ref_data.ravel()

if comp_data.ndim == 2:
    print(f"comp columns: {comp_data.shape[1]}, using mean across columns")
    comp_mean = np.nanmean(comp_data, axis=1)
else:
    comp_mean = comp_data.ravel()

print(f"ref_mean length:  {len(ref_mean)}")
print(f"comp_mean length: {len(comp_mean)}")
print(f"Length diff:      {len(ref_mean) - len(comp_mean)} samples")
print(f"Length diff pct:  {abs(len(ref_mean) - len(comp_mean)) / min(len(ref_mean), len(comp_mean)) * 100:.2f}%")

# ============================================================================
# 2. Create Profile objects
# ============================================================================
pixel_size = 3.5e-6  # meters
cutoff_hi = 1e-3     # meters (1000 um)
cutoff_lo = 5e-6     # meters (5 um)

ref_profile = Profile(
    depth_data=ref_data,
    pixel_size=pixel_size,
    cutoff_hi=cutoff_hi,
    cutoff_lo=cutoff_lo,
)

comp_profile = Profile(
    depth_data=comp_data,
    pixel_size=pixel_size,
    cutoff_hi=cutoff_hi,
    cutoff_lo=cutoff_lo,
)

params = AlignmentParameters(
    cutoff_hi=cutoff_hi,
    cutoff_lo=cutoff_lo,
)

print(f"\nPixel size: {pixel_size*1e6:.1f} um")
print(f"Cutoff hi:  {cutoff_hi*1e6:.0f} um")
print(f"Cutoff lo:  {cutoff_lo*1e6:.0f} um")

# ============================================================================
# 3. Run candidate search
# ============================================================================
print("\n" + "=" * 70)
print("3. CANDIDATE SEARCH")
print("=" * 70)

candidate_positions, shape_scales, comp_scale = find_match_candidates(
    ref_profile, comp_profile, params
)

print(f"Shape scales (um): {[s*1e6 for s in shape_scales]}")
print(f"Comp scale (um):   {comp_scale*1e6}")
print(f"Number of candidates: {len(candidate_positions)}")
print(f"Candidate positions (samples): {candidate_positions}")
print(f"Candidate positions (um):      {[p * pixel_size * 1e6 for p in candidate_positions]}")

# MATLAB expects: startPartProfile = 3.5 um => candidate at sample 1
print(f"\nExpected MATLAB candidate: sample 1 (3.5 um)")

# ============================================================================
# 4. Extract reference segment for candidate position 0
# ============================================================================
print("\n" + "=" * 70)
print("4. REFERENCE SEGMENT EXTRACTION")
print("=" * 70)

partial_length = len(comp_mean)  # 460

for cand_idx, cand_pos in enumerate(candidate_positions):
    end_idx = min(cand_pos + partial_length, len(ref_mean))
    ref_segment = ref_mean[cand_pos:end_idx]
    print(f"Candidate {cand_idx}: pos={cand_pos}, segment length={len(ref_segment)}, "
          f"range=[{np.nanmin(ref_segment):.6e}, {np.nanmax(ref_segment):.6e}]")

# Use first candidate
cand_pos_0 = candidate_positions[0] if len(candidate_positions) > 0 else 0
end_idx_0 = min(cand_pos_0 + partial_length, len(ref_mean))
ref_segment_0 = ref_mean[cand_pos_0:end_idx_0]

print(f"\nUsing candidate 0: position={cand_pos_0}, segment length={len(ref_segment_0)}")

# ============================================================================
# 5. FIRST SCALE LEVEL (500 um) - Manual alignment
# ============================================================================
print("\n" + "=" * 70)
print("5. FIRST SCALE LEVEL: 500 um (manual)")
print("=" * 70)

# The alignment function would use scale passes adjusted to <= comp_scale.
# Let's determine which scales are used:
possible_scales = np.array(list(params.scale_passes), dtype=np.float64)
resolution_limit = max(cutoff_lo, 2 * pixel_size)
print(f"Resolution limit: {resolution_limit*1e6:.1f} um")

# Filter scales the same way align_partial_profile_multiscale does
adjusted_scales = [s for s in params.scale_passes if s <= comp_scale]
print(f"Adjusted scale passes (um): {[s*1e6 for s in adjusted_scales]}")

# For this debug, use 500 um (5e-4 m) as the first scale
first_scale = 5e-4  # 500 um
if adjusted_scales:
    first_scale = adjusted_scales[0]
    print(f"Actual first scale: {first_scale*1e6:.0f} um")

# Work with equal-length profiles: ref_segment_0 and comp_mean
# They should be the same length (both 460 if candidate is valid)
work_ref = ref_segment_0.copy()
work_comp = comp_mean.copy()

# Trim to equal length if needed
min_len = min(len(work_ref), len(work_comp))
work_ref = work_ref[:min_len]
work_comp = work_comp[:min_len]
print(f"Working profiles length: {min_len}")

# Apply lowpass filter at first scale
ref_lp = _apply_lowpass_filter_1d(work_ref, first_scale, pixel_size, cut_borders=False)
comp_lp = _apply_lowpass_filter_1d(work_comp, first_scale, pixel_size, cut_borders=False)

print(f"\nAfter lowpass at {first_scale*1e6:.0f} um:")
print(f"  ref_lp range:  [{np.nanmin(ref_lp):.6e}, {np.nanmax(ref_lp):.6e}]")
print(f"  comp_lp range: [{np.nanmin(comp_lp):.6e}, {np.nanmax(comp_lp):.6e}]")
print(f"  ref_lp std:    {np.nanstd(ref_lp):.6e}")
print(f"  comp_lp std:   {np.nanstd(comp_lp):.6e}")

# Compute subsample factor (matching align_profiles_multiscale)
cutoff_samples = first_scale / pixel_size
subsample_factor = max(1, int(np.ceil(cutoff_samples / 2 / 5)))
print(f"\nCutoff in samples: {cutoff_samples:.1f}")
print(f"Subsample factor:  {subsample_factor}")

ref_sub = ref_lp[::subsample_factor]
comp_sub = comp_lp[::subsample_factor]
print(f"Subsampled lengths: ref={len(ref_sub)}, comp={len(comp_sub)}")

# Correlation at zero shift
corr_zero = compute_cross_correlation(ref_lp, comp_lp)
print(f"\nCorrelation at zero shift (full res): {corr_zero:.6f}")
corr_zero_sub = compute_cross_correlation(ref_sub, comp_sub)
print(f"Correlation at zero shift (subsampled): {corr_zero_sub:.6f}")

# Compute translation bounds (matching alignment.py logic)
max_translation_mm = params.max_translation * 1000
redetermine_max_trans = abs(max_translation_mm - 10000) < 1e-6
print(f"\nredetermine_max_trans: {redetermine_max_trans}")

if redetermine_max_trans:
    cutoff_um = first_scale * 1e6
    min_trans_adj = cutoff_um
    max_trans_adj = cutoff_um
    print(f"Translation bounds set to cutoff_um = {cutoff_um:.0f} um")
else:
    max_trans_adj = round(params.max_translation / pixel_size)
    min_trans_adj = max_trans_adj
    print(f"Translation bounds: +/- {max_trans_adj} samples")

trans_lb = -int(round(min_trans_adj / subsample_factor))
trans_ub = int(round(max_trans_adj / subsample_factor))
print(f"Subsampled translation bounds: [{trans_lb}, {trans_ub}]")

# Evaluate objective function at translations from -40 to +40 (original samples)
print(f"\n--- Objective function landscape (subsampled, first scale {first_scale*1e6:.0f} um) ---")
print(f"{'trans_orig':>12s} {'trans_sub':>12s} {'objective':>12s} {'correlation':>12s}")

scan_range_orig = range(-40, 41)
obj_values_sub = []
trans_orig_list = []

for t_orig in scan_range_orig:
    t_sub = t_orig / subsample_factor
    x_test = np.array([t_sub, 0.0])  # no scaling
    obj_val = _alignment_objective(x_test, ref_sub, comp_sub)
    obj_values_sub.append(obj_val)
    trans_orig_list.append(t_orig)
    if t_orig % 5 == 0:  # Print every 5th value
        print(f"{t_orig:12d} {t_sub:12.2f} {obj_val:12.6f} {-obj_val:12.6f}")

obj_values_sub = np.array(obj_values_sub)
best_idx = np.argmin(obj_values_sub)
best_t_orig = trans_orig_list[best_idx]
best_obj = obj_values_sub[best_idx]
print(f"\nBest translation (orig samples): {best_t_orig}")
print(f"Best objective value: {best_obj:.6f} (correlation: {-best_obj:.6f})")

# Also check a finer grid around the best
print(f"\n--- Fine grid around best ({best_t_orig}) ---")
fine_range = np.arange(best_t_orig - 5, best_t_orig + 6, 0.5)
for t_orig in fine_range:
    t_sub = t_orig / subsample_factor
    x_test = np.array([t_sub, 0.0])
    obj_val = _alignment_objective(x_test, ref_sub, comp_sub)
    print(f"  t_orig={t_orig:8.1f}, t_sub={t_sub:8.3f}, obj={obj_val:.8f}, corr={-obj_val:.8f}")

# ============================================================================
# 6. Full objective (no lowpass, no subsampling) at translations from -50 to +50
# ============================================================================
print("\n" + "=" * 70)
print("6. FULL OBJECTIVE (no lowpass, no subsampling)")
print("=" * 70)
print(f"{'translation':>12s} {'objective':>12s} {'correlation':>12s}")

obj_values_full = []
trans_full_list = list(range(-50, 51))

for t in trans_full_list:
    x_test = np.array([float(t), 0.0])
    obj_val = _alignment_objective(x_test, work_ref, work_comp)
    obj_values_full.append(obj_val)
    if t % 5 == 0:
        print(f"{t:12d} {obj_val:12.6f} {-obj_val:12.6f}")

obj_values_full = np.array(obj_values_full)
best_idx_full = np.argmin(obj_values_full)
best_t_full = trans_full_list[best_idx_full]
best_obj_full = obj_values_full[best_idx_full]
print(f"\nBest translation (full, no filter): {best_t_full}")
print(f"Best objective value: {best_obj_full:.6f} (correlation: {-best_obj_full:.6f})")

# Fine grid around best
print(f"\n--- Fine grid around best ({best_t_full}) ---")
fine_range_full = np.arange(best_t_full - 5, best_t_full + 6, 0.5)
for t in fine_range_full:
    x_test = np.array([t, 0.0])
    obj_val = _alignment_objective(x_test, work_ref, work_comp)
    print(f"  t={t:8.1f}, obj={obj_val:.8f}, corr={-obj_val:.8f}")

# ============================================================================
# 7. Summary comparison with MATLAB
# ============================================================================
print("\n" + "=" * 70)
print("7. SUMMARY: PYTHON vs MATLAB")
print("=" * 70)
print(f"MATLAB expected translation: ~35.4 samples ({-123.92:.2f} um / {pixel_size*1e6:.1f} um)")
print(f"MATLAB expected overlap:     0.924")
print(f"MATLAB expected correlation: 0.986")
print(f"MATLAB expected candidate:   sample 1 (startPartProfile=3.5 um)")
print(f"")
print(f"Python candidate position:   {cand_pos_0}")
print(f"Python best trans (1st scale, subsampled): {best_t_orig} samples")
print(f"Python best trans (full, no filter):       {best_t_full} samples")
print(f"Python corr at t=0 (full):   {-_alignment_objective(np.array([0.0, 0.0]), work_ref, work_comp):.6f}")
print(f"Python corr at t=-35 (full): {-_alignment_objective(np.array([-35.0, 0.0]), work_ref, work_comp):.6f}")
print(f"Python corr at t=+35 (full): {-_alignment_objective(np.array([35.0, 0.0]), work_ref, work_comp):.6f}")

# Also check: what if the candidate position matters?
# MATLAB says startPartProfile = 3.5 um => candidate at sample 1
# If Python found candidate at 0, the ref segment is shifted by 1 sample
if len(candidate_positions) > 0 and candidate_positions[0] != 1:
    print(f"\n--- Checking with MATLAB candidate position (sample 1) ---")
    ref_seg_matlab = ref_mean[1:1 + partial_length]
    min_len2 = min(len(ref_seg_matlab), len(comp_mean))
    ref_seg_matlab = ref_seg_matlab[:min_len2]
    comp_matlab = comp_mean[:min_len2]
    
    for t in [-40, -35, -30, -20, -10, 0, 10, 20, 30, 35, 40]:
        x_test = np.array([float(t), 0.0])
        obj_val = _alignment_objective(x_test, ref_seg_matlab, comp_matlab)
        print(f"  t={t:4d}, corr={-obj_val:.6f}")

print("\nDone.")
