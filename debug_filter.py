"""Debug script to check lowpass filter behavior and NanConv edge correction."""

import sys

sys.path.insert(0, "/Users/laurensweijs/scratch/packages/scratch-core/src")

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve as sig_convolve

# Load data
base = "/Users/laurensweijs/scratch/packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel().astype(np.float64)
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel().astype(np.float64)

pixel_size = 3.5e-6
cutoff = 500e-6  # 500 um

# Replicate MATLAB's ApplyLowPassFilter exactly
cutoff_pixels = cutoff / pixel_size  # = 142.857
sigma = cutoff_pixels * 0.187390625  # MATLAB ChebyCutoffToGaussSigma
alpha = 3.0
L = 1 + 2 * round(alpha * sigma)
L = L - 1
n = np.arange(L + 1) - L / 2
kernel = np.exp(-0.5 * (alpha * n / (L / 2)) ** 2)
kernel = kernel / np.sum(kernel)

print(f"Cutoff pixels: {cutoff_pixels}")
print(f"Sigma: {sigma}")
print(f"Kernel length: {L + 1}")

# Method 1: Python current implementation (convolve1d mode="constant")
a = ref_data[:460].copy()  # ref segment for candidate position 0
on = np.ones_like(a)

flat_scipy = convolve1d(on, kernel, mode="constant", cval=0.0)
raw_scipy = convolve1d(a, kernel, mode="constant", cval=0.0)
result_scipy = np.where(flat_scipy > 0, raw_scipy / flat_scipy, 0.0)

# Method 2: Explicit MATLAB conv2 'same' implementation
# MATLAB's conv2(a, k, 'same') is a 2D convolution with 'same' output size
# For 1D, this is equivalent to np.convolve(a, k, 'same')
# But convolve1d uses different padding convention from np.convolve

# Let's check: np.convolve with 'full' and then crop to 'same'
full_conv = np.convolve(a, kernel, mode="full")
full_flat = np.convolve(on, kernel, mode="full")

# MATLAB 'same' crops the central part of the full convolution
n_full = len(full_conv)
n_same = len(a)
start = (n_full - n_same) // 2
result_npconv = full_conv[start : start + n_same]
flat_npconv = full_flat[start : start + n_same]
result_npconv_edge = np.where(flat_npconv > 0, result_npconv / flat_npconv, 0.0)

# Method 3: scipy.signal.convolve (same mode)
result_sigconv = sig_convolve(a, kernel, mode="same")
flat_sigconv = sig_convolve(on, kernel, mode="same")
result_sigconv_edge = np.where(flat_sigconv > 0, result_sigconv / flat_sigconv, 0.0)

print("\n--- Comparison of convolution methods ---")
print(f"convolve1d vs np.convolve 'same': max diff = {np.max(np.abs(result_scipy - result_npconv_edge)):.2e}")
print(f"convolve1d vs sig.convolve 'same': max diff = {np.max(np.abs(result_scipy - result_sigconv_edge)):.2e}")
print(f"np.convolve vs sig.convolve: max diff = {np.max(np.abs(result_npconv_edge - result_sigconv_edge)):.2e}")

# Check edge values
print("\n--- Edge values (first/last 5) ---")
print(f"convolve1d result:     {result_scipy[:5]}")
print(f"np.convolve result:    {result_npconv_edge[:5]}")
print(f"sig.convolve result:   {result_sigconv_edge[:5]}")
print(f"\nconvolve1d result end: {result_scipy[-5:]}")
print(f"np.convolve result end:{result_npconv_edge[-5:]}")
print(f"sig.convolve result end:{result_sigconv_edge[-5:]}")

# Check flat values at edges
print("\n--- Flat values (edge correction) at edges ---")
print(f"convolve1d flat:     {flat_scipy[:5]} ... {flat_scipy[-5:]}")
print(f"np.convolve flat:    {flat_npconv[:5]} ... {flat_npconv[-5:]}")
print(f"sig.convolve flat:   {flat_sigconv[:5]} ... {flat_sigconv[-5:]}")

# Now check the actual impact on the objective function
# Compare lowpassed profiles from different methods
ref_seg = ref_data[:460]
comp_seg = comp_data[:460]


# Lowpass using convolve1d (current Python)
def lowpass_convolve1d(data, kern):
    a = data.copy()
    on = np.ones_like(a)
    flat = convolve1d(on, kern, mode="constant", cval=0.0)
    raw = convolve1d(a, kern, mode="constant", cval=0.0)
    return np.where(flat > 0, raw / flat, 0.0)


# Lowpass using np.convolve (MATLAB's conv2 equivalent)
def lowpass_npconv(data, kern):
    a = data.copy()
    on = np.ones_like(a)
    full_a = np.convolve(a, kern, mode="full")
    full_on = np.convolve(on, kern, mode="full")
    n_f = len(full_a)
    n_s = len(a)
    s = (n_f - n_s) // 2
    raw = full_a[s : s + n_s]
    flat = full_on[s : s + n_s]
    return np.where(flat > 0, raw / flat, 0.0)


ref_lp_1 = lowpass_convolve1d(ref_seg, kernel)
comp_lp_1 = lowpass_convolve1d(comp_seg, kernel)
ref_lp_2 = lowpass_npconv(ref_seg, kernel)
comp_lp_2 = lowpass_npconv(comp_seg, kernel)

print("\n--- Lowpassed profiles comparison ---")
print(f"ref diff (convolve1d vs npconv): max = {np.max(np.abs(ref_lp_1 - ref_lp_2)):.2e}")
print(f"comp diff (convolve1d vs npconv): max = {np.max(np.abs(comp_lp_1 - comp_lp_2)):.2e}")

# Subsample and check objective function
subsample = 15
ref_sub_1 = ref_lp_1[::subsample]
comp_sub_1 = comp_lp_1[::subsample]
ref_sub_2 = ref_lp_2[::subsample]
comp_sub_2 = comp_lp_2[::subsample]

# Correlation at t=0 for both methods
from conversion.profile_correlator.similarity import compute_cross_correlation

corr_1 = compute_cross_correlation(ref_sub_1, comp_sub_1)
corr_2 = compute_cross_correlation(ref_sub_2, comp_sub_2)
print(f"\nCorrelation at t=0 (convolve1d): {corr_1:.6f}")
print(f"Correlation at t=0 (npconv):     {corr_2:.6f}")
print(f"Difference: {abs(corr_1 - corr_2):.2e}")

# Check if the kernel is symmetric (important for conv vs corr)
print("\n--- Kernel symmetry check ---")
print(f"Kernel symmetric: {np.allclose(kernel, kernel[::-1])}")
print(f"Kernel sum: {np.sum(kernel)}")
print(f"Kernel max: {np.max(kernel)} at index {np.argmax(kernel)}")
print(f"Kernel center: index {len(kernel) // 2}")
