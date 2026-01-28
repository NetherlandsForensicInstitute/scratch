"""Check effect of NOT restoring NaN in the lowpass filter for candidate search."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.candidate_search import (
    _apply_lowpass_filter_1d,
    _resample_interpolation,
)
from conversion.profile_correlator.similarity import compute_cross_correlation
from scipy import signal

pixel_size = 3.5e-6

base = "packages/scratch-core/tests/resources/profile_correlator/partial_with_nans"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()

scale = 1e-3  # 1000um
oversampling = 16

# Method A: Normal lowpass filter (preserves NaN at original positions)
ref_filt_a = _apply_lowpass_filter_1d(ref_data, scale, pixel_size)
comp_filt_a = _apply_lowpass_filter_1d(comp_data, scale, pixel_size)


# Method B: Lowpass filter WITHOUT NaN restoration (fill in NaN with normalized convolution)
def lowpass_nonanout(profile, cutoff_wavelength, pixel_size):
    """Lowpass like candidate_search._apply_lowpass_filter_1d but WITHOUT restoring NaN."""
    from conversion.filter.gaussian import ALPHA_GAUSSIAN
    from conversion.filter.regression import apply_order0_filter, create_gaussian_kernel_1d

    profile = np.asarray(profile).ravel()
    cutoff_pixels = cutoff_wavelength / pixel_size
    has_nans = np.any(np.isnan(profile))
    kernel_1d = create_gaussian_kernel_1d(cutoff_pixels, has_nans, ALPHA_GAUSSIAN)
    kernel_identity = np.array([1.0])
    data_2d = profile[:, np.newaxis]
    mode = "constant" if has_nans else "symmetric"
    filtered_2d = apply_order0_filter(data_2d, kernel_identity, kernel_1d, mode=mode)
    # NOTE: Do NOT restore NaN positions
    return filtered_2d.ravel()


ref_filt_b = lowpass_nonanout(ref_data, scale, pixel_size)
comp_filt_b = lowpass_nonanout(comp_data, scale, pixel_size)

print(f"Method A (nanout): ref NaN={np.sum(np.isnan(ref_filt_a))}, comp NaN={np.sum(np.isnan(comp_filt_a))}")
print(f"Method B (nonanout): ref NaN={np.sum(np.isnan(ref_filt_b))}, comp NaN={np.sum(np.isnan(comp_filt_b))}")

# Resample both
scale_samples = scale / pixel_size
subsampling = max(1, int(np.floor(scale_samples / oversampling)))
zoom = 1.0 / subsampling
print(f"subsampling={subsampling}")

ref_sub_a = _resample_interpolation(ref_filt_a, zoom)
comp_sub_a = _resample_interpolation(comp_filt_a, zoom)
ref_sub_b = _resample_interpolation(ref_filt_b, zoom)
comp_sub_b = _resample_interpolation(comp_filt_b, zoom)

# Also try: FFT resample with nonanout (no NaN to worry about)
ref_len_sub = max(1, int(round(len(ref_filt_b) / subsampling)))
comp_len_sub = max(1, int(round(len(comp_filt_b) / subsampling)))
ref_sub_c = np.asarray(signal.resample(ref_filt_b, ref_len_sub), dtype=np.float64)
comp_sub_c = np.asarray(signal.resample(comp_filt_b, comp_len_sub), dtype=np.float64)

print(f"\nA (nanout+bspline): ref_sub NaN={np.sum(np.isnan(ref_sub_a))}, comp_sub NaN={np.sum(np.isnan(comp_sub_a))}")
print(f"B (nonanout+bspline): ref_sub NaN={np.sum(np.isnan(ref_sub_b))}, comp_sub NaN={np.sum(np.isnan(comp_sub_b))}")
print(f"C (nonanout+FFT): ref_sub NaN={np.sum(np.isnan(ref_sub_c))}, comp_sub NaN={np.sum(np.isnan(comp_sub_c))}")

for name, ref_sub, comp_sub in [
    ("A (nanout+bspline)", ref_sub_a, comp_sub_a),
    ("B (nonanout+bspline)", ref_sub_b, comp_sub_b),
    ("C (nonanout+FFT)", ref_sub_c, comp_sub_c),
]:
    n_pos = max(1, len(ref_sub) - len(comp_sub) + 1)
    print(f"\n{name}: ref={len(ref_sub)}, comp={len(comp_sub)}, positions={n_pos}")
    for pos in range(n_pos):
        end_pos = pos + len(comp_sub)
        ref_seg = ref_sub[pos:end_pos]
        min_len = min(len(ref_seg), len(comp_sub))
        xcorr = compute_cross_correlation(ref_seg[:min_len], comp_sub[:min_len])
        above = "✓" if xcorr >= 0.5 else " "
        print(f"  pos={pos:2d}: xcorr={xcorr:+.4f} {above}")
