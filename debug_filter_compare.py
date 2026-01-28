"""Compare Python's lowpass filter with a MATLAB-simulated conv2 implementation."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve as signal_convolve

# Load edge_over_threshold data
base = "packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()
pixel_size = 3.5e-6

# Use the first 460 samples (candidate 0)
profile = ref_data[:460].copy()

# Compute kernel for 500um cutoff
cutoff = 500e-6
cutoff_pixels = cutoff / pixel_size  # 142.857
sigma = cutoff_pixels * 0.187390625  # 26.770...
alpha = 3.0
L = 1 + 2 * round(alpha * sigma)
L = L - 1
n = np.arange(L + 1) - L / 2
kernel = np.exp(-0.5 * (alpha * n / (L / 2)) ** 2)
kernel = kernel / np.sum(kernel)

print(f"cutoff_pixels={cutoff_pixels:.4f}, sigma={sigma:.4f}, L={L}, kernel_size={len(kernel)}")
print(f"kernel sum={np.sum(kernel):.15f}")
print(f"kernel first 5: {kernel[:5]}")
print(f"kernel center±2: {kernel[len(kernel) // 2 - 2 : len(kernel) // 2 + 3]}")

# No NaN in edge_over_threshold data, so NanConv simplifies to:
# In MATLAB: NanConv(profile, t, 'nanout', 'edge') with no NaN
#   n = isnan(a) → all False
#   a(n) = 0 → no-op
#   on(n) = 0 → no-op
#   flat = conv2(on, k, 'same') → conv2(ones, k, 'same')
#   c = conv2(a, k, 'same') / flat
# Since no NaN and 'edge' is specified:
#   flat = conv2(ones, kernel, 'same')
#   result = conv2(profile, kernel, 'same') / flat

# Method 1: Python's current implementation (convolve1d)
on = np.ones_like(profile)
flat_method1 = convolve1d(on, kernel, mode="constant", cval=0.0)
raw_method1 = convolve1d(profile, kernel, mode="constant", cval=0.0)
result_method1 = np.where(flat_method1 > 0, raw_method1 / flat_method1, 0.0)


# Method 2: Using scipy.signal.convolve (mimics MATLAB conv2)
def conv2_same_1d(a, k):
    """Simulate MATLAB's conv2(a, k, 'same') for 1D vectors."""
    full = signal_convolve(a, k, mode="full")
    # Extract 'same' portion
    n = len(a)
    m = len(k)
    start = m // 2
    return full[start : start + n]


flat_method2 = conv2_same_1d(on, kernel)
raw_method2 = conv2_same_1d(profile, kernel)
result_method2 = np.where(flat_method2 > 0, raw_method2 / flat_method2, 0.0)


# Method 3: numpy.convolve
def np_conv_same(a, k):
    """numpy.convolve with 'same' output."""
    full = np.convolve(a, k, mode="full")
    n = len(a)
    m = len(k)
    start = m // 2
    return full[start : start + n]


flat_method3 = np_conv_same(on, kernel)
raw_method3 = np_conv_same(profile, kernel)
result_method3 = np.where(flat_method3 > 0, raw_method3 / flat_method3, 0.0)

# Compare results
print("\n=== FILTER COMPARISON ===")
print(f"Profile length: {len(profile)}")

# Check if methods produce different results
diff_12 = np.max(np.abs(result_method1 - result_method2))
diff_13 = np.max(np.abs(result_method1 - result_method3))
diff_23 = np.max(np.abs(result_method2 - result_method3))

print("\nMax absolute differences:")
print(f"  convolve1d vs signal.convolve: {diff_12:.2e}")
print(f"  convolve1d vs numpy.convolve:  {diff_13:.2e}")
print(f"  signal.convolve vs numpy:      {diff_23:.2e}")

# Check raw convolution differences (before normalization)
diff_raw_12 = np.max(np.abs(raw_method1 - raw_method2))
diff_flat_12 = np.max(np.abs(flat_method1 - flat_method2))
print("\nRaw convolution max diff (convolve1d vs signal.convolve):")
print(f"  raw: {diff_raw_12:.2e}")
print(f"  flat: {diff_flat_12:.2e}")

# Show values at boundaries and center
for idx in [0, 1, 2, 79, 80, 81, len(profile) // 2, len(profile) - 3, len(profile) - 2, len(profile) - 1]:
    print(f"\nIndex {idx}:")
    print(f"  convolve1d:      {result_method1[idx]:.15e}")
    print(f"  signal.convolve: {result_method2[idx]:.15e}")
    print(f"  numpy.convolve:  {result_method3[idx]:.15e}")
    if abs(result_method1[idx] - result_method2[idx]) > 1e-16:
        print(f"  DIFF: {result_method1[idx] - result_method2[idx]:.2e}")

# Check flat values at boundaries (this is the edge correction divisor)
print("\n=== FLAT (edge correction) ===")
for idx in [0, 1, 79, 80, len(profile) - 81, len(profile) - 80, len(profile) - 2, len(profile) - 1]:
    print(f"  flat[{idx}]: convolve1d={flat_method1[idx]:.15f}, signal={flat_method2[idx]:.15f}")

# Now check if the difference matters for correlation
from conversion.profile_correlator.similarity import compute_cross_correlation

# Subsample by 15 (for 500um scale)
subsample_factor = 15
p1_sub_m1 = result_method1[::subsample_factor]
p2_sub_comp_m1 = convolve1d(comp_data, kernel, mode="constant", cval=0.0)
flat_comp_m1 = convolve1d(np.ones_like(comp_data), kernel, mode="constant", cval=0.0)
p2_filt_m1 = np.where(flat_comp_m1 > 0, p2_sub_comp_m1 / flat_comp_m1, 0.0)
p2_sub_m1 = p2_filt_m1[::subsample_factor]

p1_sub_m2 = result_method2[::subsample_factor]
p2_sub_comp_m2 = conv2_same_1d(comp_data, kernel)
flat_comp_m2 = conv2_same_1d(np.ones_like(comp_data), kernel)
p2_filt_m2 = np.where(flat_comp_m2 > 0, p2_sub_comp_m2 / flat_comp_m2, 0.0)
p2_sub_m2 = p2_filt_m2[::subsample_factor]

corr_m1 = compute_cross_correlation(p1_sub_m1, p2_sub_m1)
corr_m2 = compute_cross_correlation(p1_sub_m2, p2_sub_m2)

print("\n=== CORRELATION COMPARISON (subsampled) ===")
print(f"  convolve1d:      corr={corr_m1:.15f}")
print(f"  signal.convolve: corr={corr_m2:.15f}")
print(f"  difference: {corr_m1 - corr_m2:.2e}")
print(f"  subsampled lengths: {len(p1_sub_m1)}, {len(p2_sub_m1)}")
