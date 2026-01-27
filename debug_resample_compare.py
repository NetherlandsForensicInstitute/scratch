"""Compare resampling methods' effect on candidate correlations."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.candidate_search import _apply_lowpass_filter_1d
from conversion.profile_correlator.similarity import compute_cross_correlation
from scipy import signal

pixel_size = 3.5e-6

base = "packages/scratch-core/tests/resources/profile_correlator/partial_with_nans"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()

scale = 1e-3  # 1000um
ref_filt = _apply_lowpass_filter_1d(ref_data, scale, pixel_size)
comp_filt = _apply_lowpass_filter_1d(comp_data, scale, pixel_size)

scale_samples = scale / pixel_size
subsampling = max(1, int(np.floor(scale_samples / 16)))
print(f"subsampling = {subsampling}")
print(f"ref_filt: len={len(ref_filt)}, NaN={np.sum(np.isnan(ref_filt))}")
print(f"comp_filt: len={len(comp_filt)}, NaN={np.sum(np.isnan(comp_filt))}")

# Method 1: FFT-based with NaN→0 (original fix)
ref_clean = np.where(np.isnan(ref_filt), 0.0, ref_filt)
comp_clean = np.where(np.isnan(comp_filt), 0.0, comp_filt)
ref_len_sub = max(1, int(round(len(ref_filt) / subsampling)))
comp_len_sub = max(1, int(round(len(comp_filt) / subsampling)))
ref_sub_fft = np.asarray(signal.resample(ref_clean, ref_len_sub), dtype=np.float64)
comp_sub_fft = np.asarray(signal.resample(comp_clean, comp_len_sub), dtype=np.float64)

# Method 2: FFT-based NaN mask normalization
ref_mask = np.where(np.isnan(ref_filt), 0.0, 1.0)
comp_mask = np.where(np.isnan(comp_filt), 0.0, 1.0)
ref_sub_data = np.asarray(signal.resample(ref_clean, ref_len_sub), dtype=np.float64)
ref_sub_mask = np.asarray(signal.resample(ref_mask, ref_len_sub), dtype=np.float64)
comp_sub_data = np.asarray(signal.resample(comp_clean, comp_len_sub), dtype=np.float64)
comp_sub_mask = np.asarray(signal.resample(comp_mask, comp_len_sub), dtype=np.float64)
ref_sub_norm = np.where(ref_sub_mask > 0.5, ref_sub_data / ref_sub_mask, np.nan)
comp_sub_norm = np.where(comp_sub_mask > 0.5, comp_sub_data / comp_sub_mask, np.nan)


# Method 3: Linear interpolation with NaN normalization (current)
def interp_resample(data, zoom):
    n_in = len(data)
    n_out = max(1, int(round(n_in * zoom)))
    new_x = np.arange(n_out, dtype=np.float64) / zoom
    old_x = np.arange(n_in, dtype=np.float64)
    new_x = np.clip(new_x, 0.0, n_in - 1.0)
    has_nans = np.any(np.isnan(data))
    if has_nans:
        mask = np.where(np.isnan(data), 0.0, 1.0)
        data_c = np.where(np.isnan(data), 0.0, data)
        dr = np.interp(new_x, old_x, data_c)
        mr = np.interp(new_x, old_x, mask)
        return np.where(mr > 0.5, dr / mr, np.nan)
    return np.interp(new_x, old_x, data)


zoom = 1.0 / subsampling
ref_sub_interp = interp_resample(ref_filt, zoom)
comp_sub_interp = interp_resample(comp_filt, zoom)

print(
    f"\nMethod 1 (FFT+NaN→0): ref_sub NaN={np.sum(np.isnan(ref_sub_fft))}, comp_sub NaN={np.sum(np.isnan(comp_sub_fft))}"
)
print(
    f"Method 2 (FFT+mask):   ref_sub NaN={np.sum(np.isnan(ref_sub_norm))}, comp_sub NaN={np.sum(np.isnan(comp_sub_norm))}"
)
print(
    f"Method 3 (interp+mask): ref_sub NaN={np.sum(np.isnan(ref_sub_interp))}, comp_sub NaN={np.sum(np.isnan(comp_sub_interp))}"
)

print("\nCorrelations at each position:")
print(f"{'pos':>3}  {'FFT+NaN0':>10}  {'FFT+mask':>10}  {'interp':>10}")
for method_name, ref_sub, comp_sub in [
    ("FFT+NaN0", ref_sub_fft, comp_sub_fft),
    ("FFT+mask", ref_sub_norm, comp_sub_norm),
    ("interp", ref_sub_interp, comp_sub_interp),
]:
    n_pos = max(1, len(ref_sub) - len(comp_sub) + 1)
    print(f"\n{method_name}: ref={len(ref_sub)}, comp={len(comp_sub)}, positions={n_pos}")
    for pos in range(n_pos):
        end_pos = pos + len(comp_sub)
        ref_seg = ref_sub[pos:end_pos]
        min_len = min(len(ref_seg), len(comp_sub))
        xcorr = compute_cross_correlation(ref_seg[:min_len], comp_sub[:min_len])
        above = "✓" if xcorr >= 0.5 else " "
        print(f"  pos={pos:2d}: xcorr={xcorr:+.4f} {above}")
