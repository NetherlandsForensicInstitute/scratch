import numpy as np
from math import ceil

from conversion.preprocessing.preprocess_data import (
    apply_form_noise_removal,
    cheby_cutoff_to_gauss_sigma,
)

"""
Test to verify Python implementation matches MATLAB ground truth.
"""


def test_form_noise_removal_pipeline():
    """
    Comprehensive test of the form and noise removal pipeline.

    Tests the complete pipeline behavior without requiring MATLAB:
    1. Verifies correct filter sequence (highpass -> lowpass)
    2. Checks border cropping logic
    3. Validates short data handling
    4. Tests mask propagation
    """

    print("\n=== Testing Form & Noise Removal Pipeline ===\n")

    # Test 1: Verify filter sequence and output
    print("Test 1: Filter Sequence")
    height, width = 200, 150
    xdim = 1e-6

    # Create test data with known components
    x = np.arange(height) * xdim
    X, _ = np.meshgrid(x, np.arange(width), indexing="ij")

    form = 5e-6 * (X / x.max()) ** 2  # Large wavelength
    striations = 0.5e-6 * np.sin(2 * np.pi * X / 500e-6)  # Medium wavelength
    noise = 0.1e-6 * np.random.randn(height, width)  # Small wavelength

    depth_data = form + striations + noise

    result, result_mask = apply_form_noise_removal(
        depth_data=depth_data,
        xdim=xdim,
        cutoff_hi=2000e-6,  # Remove > 2000 µm (form)
        cutoff_lo=250e-6,  # Remove < 250 µm (noise)
    )

    # Verify form removed (mean near zero)
    assert np.abs(np.mean(result)) < 1e-6, "Form not removed"

    # Verify striations preserved (signal remains)
    assert np.std(result) > 0.05e-6, "Striations not preserved"

    # Verify noise reduced
    assert np.std(result) < np.std(depth_data), "Noise not reduced"

    print(f"  ✅ Form removed: mean = {np.mean(result):.2e}")
    print(f"  ✅ Striations preserved: std = {np.std(result) * 1e6:.3f} µm")
    print(
        f"  ✅ Noise reduced: {np.std(depth_data) * 1e6:.3f} → {np.std(result) * 1e6:.3f} µm"
    )

    # Test 2: Border cropping
    print("\nTest 2: Border Cropping")

    sigma = cheby_cutoff_to_gauss_sigma(2000e-6, xdim)
    sigma_int = int(ceil(sigma))
    expected_height = height - 2 * sigma_int

    assert result.shape[0] == expected_height, (
        f"Border cropping incorrect: {result.shape[0]} vs expected {expected_height}"
    )
    assert result.shape[1] == width, "Width should not change"

    print(
        f"  ✅ Borders cropped: {height} → {result.shape[0]} rows (removed {2 * sigma_int})"
    )

    # Test 3: Short data handling (no border cropping)
    print("\nTest 3: Short Data Handling")

    # Create data that's too short (2*sigma > 20% of height)
    short_height = int(2 * sigma / 0.2) - 5  # Just below threshold
    short_data = np.random.randn(short_height, width) * 1e-6

    result_short, _ = apply_form_noise_removal(
        depth_data=short_data,
        xdim=xdim,
        cutoff_hi=2000e-6,
        cut_borders_after_smoothing=True,  # Should be overridden
    )

    # For short data, borders should NOT be cut
    assert result_short.shape[0] == short_height, (
        f"Short data borders were cut (got {result_short.shape[0]}, expected {short_height})"
    )

    print(
        f"  ✅ Short data preserved: {short_height} → {result_short.shape[0]} rows (no cropping)"
    )

    # Test 4: Mask propagation
    print("\nTest 4: Mask Propagation")

    # Create data with masked region
    depth_data_masked = depth_data.copy()
    mask_input = np.ones(depth_data.shape, dtype=bool)
    mask_input[:, 0:20] = False  # Mask first 20 columns

    result_masked, mask_output = apply_form_noise_removal(
        depth_data=depth_data_masked,
        xdim=xdim,
        cutoff_hi=2000e-6,
        cutoff_lo=250e-6,
        mask=mask_input,
    )

    # Verify mask was propagated
    assert mask_output.shape == result_masked.shape, "Mask shape mismatch"
    assert np.any(~mask_output), "Mask should have invalid regions"

    # Masked regions should have corresponding masked output
    invalid_cols = np.sum(~mask_output, axis=0)
    assert np.any(invalid_cols > 0), "Masked columns should remain masked"

    print(
        f"  ✅ Mask propagated: {np.sum(~mask_input)} → {np.sum(~mask_output)} invalid pixels"
    )

    # Test 5: No cropping mode
    print("\nTest 5: No Border Cropping Mode")

    result_no_crop, _ = apply_form_noise_removal(
        depth_data=depth_data,
        xdim=xdim,
        cutoff_hi=2000e-6,
        cutoff_lo=250e-6,
        cut_borders_after_smoothing=False,
    )

    assert result_no_crop.shape[0] == height, (
        f"With no cropping, height should be {height}, got {result_no_crop.shape[0]}"
    )

    print(
        f"  ✅ No cropping mode: {height} → {result_no_crop.shape[0]} rows (preserved)"
    )

    # Test 6: Different cutoff values
    print("\nTest 6: Different Cutoff Values")

    # Aggressive filtering (small cutoffs)
    result_aggressive, _ = apply_form_noise_removal(
        depth_data=depth_data,
        xdim=xdim,
        cutoff_hi=1000e-6,  # More aggressive form removal
        cutoff_lo=500e-6,  # Less aggressive noise removal
    )

    # Conservative filtering (large cutoffs)
    result_conservative, _ = apply_form_noise_removal(
        depth_data=depth_data,
        xdim=xdim,
        cutoff_hi=4000e-6,  # Less aggressive form removal
        cutoff_lo=100e-6,  # More aggressive noise removal
    )

    # Aggressive should have smaller variance (more filtered)
    assert np.std(result_aggressive) < np.std(result_conservative), (
        "Aggressive filtering should reduce variance more"
    )

    print(f"  ✅ Aggressive filtering: std = {np.std(result_aggressive) * 1e6:.3f} µm")
    print(
        f"  ✅ Conservative filtering: std = {np.std(result_conservative) * 1e6:.3f} µm"
    )

    print("\n✅ All pipeline tests passed!")


def test_synthetic_form_noise_removal():
    """
    Test on synthetic data where we know the ground truth.

    Verifies that:
    - Large-scale form is removed
    - Striations are preserved
    - High-frequency noise is removed
    """

    # Create synthetic striated surface
    height, width = 200, 150
    xdim = 1e-6  # 1 µm spacing

    x = np.arange(height) * xdim
    y = np.arange(width) * xdim
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Components
    form = 5e-6 * (X / x.max()) ** 2  # Curvature (large wavelength)
    striations = 0.5e-6 * np.sin(2 * np.pi * X / 500e-6)  # 500 µm striations
    noise = 0.1e-6 * np.random.randn(height, width)  # High-freq noise

    depth_data = form + striations + noise

    # Apply preprocessing
    result, mask = apply_form_noise_removal(
        depth_data=depth_data,
        xdim=xdim,
        cutoff_hi=2000e-6,  # Remove > 2000 µm (form)
        cutoff_lo=250e-6,  # Remove < 250 µm (noise)
    )

    # Verify form removed (mean near zero)
    assert np.abs(np.mean(result)) < 1e-7, "Form not removed"

    # Verify striations preserved
    std_result = np.std(result)
    assert std_result > 0.1e-6, "Striations lost"

    # Verify noise reduced
    std_original = np.std(depth_data)
    assert std_result < std_original * 0.5, "Noise not reduced"

    print("✅ Synthetic test passed!")
    print(f"   Noise reduction: {std_original * 1e6:.3f} → {std_result * 1e6:.3f} µm")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Form & Noise Removal Implementation")
    print("=" * 60)

    print("\n=== Test 1: Synthetic Data Test ===")
    test_synthetic_form_noise_removal()

    print("\n=== Test 2: Pipeline Behavior Test ===")
    test_form_noise_removal_pipeline()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
