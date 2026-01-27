"""Trace candidate search with threshold lowering for partial_with_nans."""
import sys
sys.path.insert(0, 'packages/scratch-core/src')

import numpy as np
from conversion.profile_correlator.data_types import Profile, AlignmentParameters
from conversion.profile_correlator.candidate_search import find_match_candidates

pixel_size = 3.5e-6

base = 'packages/scratch-core/tests/resources/profile_correlator/partial_with_nans'
ref_data = np.load(f'{base}/input_profile_ref.npy').ravel()
comp_data = np.load(f'{base}/input_profile_comp.npy').ravel()

pr = Profile(depth_data=ref_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)
pc = Profile(depth_data=comp_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)

# Try different thresholds
for threshold in [0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0, -0.5, -1.0]:
    params = AlignmentParameters(
        cutoff_hi=1e-3, cutoff_lo=5e-6,
        inclusion_threshold=threshold,
    )
    pos, shape_scales, comp_scale = find_match_candidates(pr, pc, params)
    print(f"threshold={threshold:+.2f}: candidates={pos.tolist()}, n_candidates={len(pos)}")
    if len(pos) > 0:
        break

# Also check: what are the actual correlation values at each position?
print("\n--- Checking raw correlations at 1000um scale ---")
from conversion.profile_correlator.candidate_search import (
    _apply_lowpass_filter_1d, _resample_interpolation,
)
from conversion.profile_correlator.similarity import compute_cross_correlation

ref_mean = ref_data if ref_data.ndim == 1 else np.nanmean(ref_data, axis=1)
comp_mean = comp_data if comp_data.ndim == 1 else np.nanmean(comp_data, axis=1)

scale = 1e-3  # 1000um
ref_filt = _apply_lowpass_filter_1d(ref_mean, scale, pixel_size)
comp_filt = _apply_lowpass_filter_1d(comp_mean, scale, pixel_size)

scale_samples = scale / pixel_size
subsampling = max(1, int(np.floor(scale_samples / 16)))
print(f"subsampling = {subsampling}")

zoom = 1.0 / subsampling
ref_sub = _resample_interpolation(ref_filt, zoom)
comp_sub = _resample_interpolation(comp_filt, zoom)

print(f"ref_sub: len={len(ref_sub)}, NaN={np.sum(np.isnan(ref_sub))}")
print(f"comp_sub: len={len(comp_sub)}, NaN={np.sum(np.isnan(comp_sub))}")

n_pos = max(1, len(ref_sub) - len(comp_sub) + 1)
print(f"n_positions: {n_pos}")

for pos in range(n_pos):
    end_pos = pos + len(comp_sub)
    ref_seg = ref_sub[pos:end_pos]
    min_len = min(len(ref_seg), len(comp_sub))
    xcorr = compute_cross_correlation(ref_seg[:min_len], comp_sub[:min_len])
    n_nan_ref = np.sum(np.isnan(ref_seg[:min_len]))
    n_nan_comp = np.sum(np.isnan(comp_sub[:min_len]))
    print(f"  pos={pos}: xcorr={xcorr:.4f}, ref_nans={n_nan_ref}, comp_nans={n_nan_comp}")
