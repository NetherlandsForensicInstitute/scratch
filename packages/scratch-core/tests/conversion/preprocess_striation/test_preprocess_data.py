"""
Tests for preprocess_data.py and preprocess_data_filter.py.
One test per function.
"""

import numpy as np
import pytest
from math import ceil

from conversion.preprocess_striation.preprocess_data_filter import (
    _apply_nan_weighted_gaussian_1d,
    _remove_zero_border,
    apply_gaussian_filter_1d,
    cheby_cutoff_to_gauss_sigma,
)
from conversion.preprocess_striation.preprocess_data import (
    apply_shape_noise_removal,
)
from container_models.scan_image import ScanImage


# =============================================================================
# Tests for preprocess_data_filter.py
# =============================================================================


def test_cheby_cutoff_to_gauss_sigma():
    """Test conversion from cutoff wavelength to Gaussian sigma."""
    cutoff = 1000e-6  # 1000 µm
    pixel_size = 10e-6  # 10 µm per pixel

    sigma = cheby_cutoff_to_gauss_sigma(cutoff, pixel_size)

    # The function uses: alpha_gaussian = sqrt(2 * log(2)) / (2 * pi)
    alpha_gaussian = np.sqrt(2 * np.log(2)) / (2 * np.pi)
    expected = (cutoff / pixel_size) * alpha_gaussian
    assert sigma == pytest.approx(expected)


def test_apply_nan_weighted_gaussian_1d():
    """Test that NaN values are handled correctly in 1D Gaussian filter."""
    data = np.ones((20, 10), dtype=float)
    data[10, 5] = np.nan  # Single NaN value

    result = _apply_nan_weighted_gaussian_1d(data, sigma=2.0, truncate=4.0)

    # Result should not have NaN propagation (NaN is interpolated)
    assert not np.isnan(result[10, 5])
    # Non-NaN regions should remain close to 1.0
    assert result[0, 0] == pytest.approx(1.0, rel=0.1)


def test_remove_zero_border():
    """Test that zero borders are correctly removed."""
    data = np.zeros((10, 10), dtype=float)
    mask = np.zeros((10, 10), dtype=bool)

    # Valid data only in center region
    data[3:7, 2:8] = 1.0
    mask[3:7, 2:8] = True

    cropped_data, cropped_mask, range_indices = _remove_zero_border(data, mask)

    assert cropped_data.shape == (4, 6)
    assert cropped_mask.shape == (4, 6)
    assert np.array_equal(range_indices, np.arange(3, 7))


def test_apply_gaussian_filter_1d_lowpass():
    """Test lowpass filtering removes high-frequency noise."""
    np.random.seed(42)

    # Create signal with low-frequency component + high-frequency noise
    rows = np.linspace(0, 10, 100)
    low_freq = np.sin(2 * np.pi * rows / 10)  # Low frequency signal
    high_freq = np.random.randn(100) * 0.5  # High frequency noise

    surface = np.tile((low_freq + high_freq).reshape(-1, 1), (1, 20))

    scan_image = ScanImage(data=surface, scale_x=1e-6, scale_y=1e-6)
    smoothed, indices, mask = apply_gaussian_filter_1d(
        scan_image=scan_image,
        cutoff=250e-6,
        is_high_pass=False,
        cut_borders_after_smoothing=False,
    )

    # Smoothed data should have lower variance than input
    assert np.std(smoothed) < np.std(surface)


def test_apply_gaussian_filter_1d_highpass():
    """Test highpass filtering removes low-frequency shape."""
    # Create surface with parabolic shape + fine detail
    rows = np.linspace(0, 100, 200)
    shape = 0.01 * (rows - 50) ** 2  # Parabolic shape (range ~25)
    detail = np.sin(2 * np.pi * rows / 5) * 0.1  # Fine detail (range ~0.2)

    surface = np.tile((shape + detail).reshape(-1, 1), (1, 20))

    # Use pixel size that makes cutoff effective for shape removal
    scan_image = ScanImage(data=surface, scale_x=1e-3, scale_y=1e-3)
    residuals, indices, mask = apply_gaussian_filter_1d(
        scan_image=scan_image,
        cutoff=50e-3,
        is_high_pass=True,
        cut_borders_after_smoothing=False,
    )

    # Residuals should have much smaller range than input (shape removed)
    assert np.ptp(residuals) < np.ptp(surface) * 0.25


# =============================================================================
# Tests for preprocess_data.py
# =============================================================================


def test_apply_shape_noise_removal():
    """Test complete form and noise removal pipeline."""
    np.random.seed(42)

    # Create synthetic surface: shape + striations + noise
    rows = np.linspace(0, 100, 300)
    shape = 0.001 * (rows - 50) ** 2  # Parabolic form
    striations = np.sin(2 * np.pi * rows / 10) * 0.01  # Striation pattern
    noise = np.random.randn(300) * 0.001  # Fine noise

    surface = np.tile((shape + striations + noise).reshape(-1, 1), (1, 50))

    scan_image = ScanImage(data=surface, scale_x=1e-6, scale_y=1e-6)
    result, mask = apply_shape_noise_removal(
        scan_image=scan_image,
        highpass_cutoff=2000e-6,
        lowpass_cutoff=250e-6,
    )

    # Result should have form removed (smaller range than input)
    assert np.ptp(result) < np.ptp(surface)
    # Mask should be all True (no invalid regions)
    assert mask.all()


def test_form_noise_removal_pipeline():
    """
    Comprehensive test of the form and noise removal pipeline.

    Tests:
    1. Verifies correct filter sequence (highpass -> lowpass)
    2. Checks border cropping logic
    3. Validates short data handling
    4. Tests mask propagation
    """
    np.random.seed(42)

    # Test 1: Verify filter sequence and output
    height, width = 200, 150
    scale = 1e-6

    # Create test data with known components
    x = np.arange(height) * scale
    X, _ = np.meshgrid(x, np.arange(width), indexing="ij")

    form = 5e-6 * (X / x.max()) ** 2  # Large wavelength
    striations = 0.5e-6 * np.sin(2 * np.pi * X / 500e-6)  # Medium wavelength
    noise = 0.1e-6 * np.random.randn(height, width)  # Small wavelength

    depth_data = form + striations + noise

    scan_image = ScanImage(data=depth_data, scale_x=scale, scale_y=scale)
    result, _ = apply_shape_noise_removal(
        scan_image=scan_image,
        highpass_cutoff=2000e-6,
        lowpass_cutoff=250e-6,
    )

    # Verify form removed (mean near zero)
    assert np.abs(np.mean(result)) < 1e-6, "Form not removed"

    # Verify striations preserved (signal remains)
    assert np.std(result) > 0.05e-6, "Striations not preserved"

    # Verify noise reduced
    assert np.std(result) < np.std(depth_data), "Noise not reduced"

    # Test 2: Border cropping behavior
    sigma = cheby_cutoff_to_gauss_sigma(2000e-6, scale)
    data_too_short = (2 * sigma) > (height * 0.2)

    if data_too_short:
        expected_height = height
    else:
        sigma_int = int(ceil(sigma))
        expected_height = height - 2 * sigma_int

    assert result.shape[0] == expected_height, (
        f"Border cropping incorrect: {result.shape[0]} vs expected {expected_height}"
    )
    assert result.shape[1] == width, "Width should not change"

    # Test 3: Short data handling (no border cropping due to data size)
    short_height = int(2 * sigma / 0.2) - 5
    short_data = np.random.randn(short_height, width) * 1e-6

    short_scan_image = ScanImage(data=short_data, scale_x=scale, scale_y=scale)
    result_short, _ = apply_shape_noise_removal(
        scan_image=short_scan_image,
        highpass_cutoff=2000e-6,
    )

    # Short data should not have borders cut (automatic detection)
    assert result_short.shape[0] == short_height, (
        f"Short data borders were cut (got {result_short.shape[0]}, expected {short_height})"
    )

    # Test 4: Mask propagation
    mask_input = np.ones(depth_data.shape, dtype=bool)
    mask_input[:, 0:20] = False

    masked_scan_image = ScanImage(data=depth_data.copy(), scale_x=scale, scale_y=scale)
    result_masked, mask_output = apply_shape_noise_removal(
        scan_image=masked_scan_image,
        highpass_cutoff=2000e-6,
        lowpass_cutoff=250e-6,
        mask=mask_input,
    )

    assert mask_output.shape == result_masked.shape, "Mask shape mismatch"
    assert np.any(~mask_output), "Mask should have invalid regions"


def test_synthetic_form_noise_removal():
    """
    Test on synthetic data where we know the ground truth.

    Verifies that:
    - Large-scale form is removed
    - Striations are preserved
    - High-frequency noise is removed
    """
    np.random.seed(42)

    height, width = 200, 150
    scale = 1e-6

    x = np.arange(height) * scale
    y = np.arange(width) * scale
    X, _ = np.meshgrid(x, y, indexing="ij")

    form = 5e-6 * (X / x.max()) ** 2
    striations = 0.5e-6 * np.sin(2 * np.pi * X / 500e-6)
    noise = 0.1e-6 * np.random.randn(height, width)

    depth_data = form + striations + noise

    scan_image = ScanImage(data=depth_data, scale_x=scale, scale_y=scale)
    result, _ = apply_shape_noise_removal(
        scan_image=scan_image,
        highpass_cutoff=2000e-6,
        lowpass_cutoff=250e-6,
    )

    # Verify form removed (mean near zero)
    assert np.abs(np.mean(result)) < 1e-6, "Form not removed"

    # Verify striations preserved
    std_result = np.std(result)
    assert std_result > 0.1e-6, "Striations lost"

    # Verify noise reduced (result std should be less than original)
    std_original = np.std(depth_data)
    assert std_result < std_original, "Noise not reduced"
