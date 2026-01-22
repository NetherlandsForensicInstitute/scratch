"""
Tests for preprocess_striation.py and related filter functions.
"""

import numpy as np
import pytest
from math import ceil

from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType
from conversion.filter import (
    _apply_nan_weighted_gaussian_1d,
    _remove_zero_border,
    apply_striation_preserving_filter_1d,
    cutoff_to_gaussian_sigma,
)
from conversion.preprocess_striation.parameters import PreprocessingStriationParams
from conversion.preprocess_striation.preprocess_striation import (
    _shear_data_by_shifting_profiles,
    _rotate_image_grad_vector,
    apply_shape_noise_removal,
    extract_profile,
    fine_align_bullet_marks,
    preprocess_data,
)
from conversion.resample import resample_scan_image_and_mask


# =============================================================================
# Tests for filter.py utility functions
# =============================================================================


def test_cutoff_to_gaussian_sigma():
    """Test conversion from cutoff wavelength to Gaussian sigma."""
    cutoff = 1000e-6  # 1000 µm
    pixel_size = 10e-6  # 10 µm per pixel

    sigma = cutoff_to_gaussian_sigma(cutoff, pixel_size)

    # The function uses: σ = α·λc/√(2π) where α = √(ln(2)/π)
    alpha_gaussian = np.sqrt(np.log(2) / np.pi)
    expected = alpha_gaussian * (cutoff / pixel_size) / np.sqrt(2 * np.pi)
    assert sigma == pytest.approx(expected)


def test_apply_nan_weighted_gaussian_1d():
    """Test that NaN values are handled correctly in 1D Gaussian filter."""
    data = np.ones((20, 10), dtype=float)
    data[10, 5] = np.nan  # Single NaN value

    result = _apply_nan_weighted_gaussian_1d(
        data, cutoff_length=10.0, pixel_size=1.0, axis=0
    )

    # NaN positions should be preserved
    assert np.isnan(result[10, 5])
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


def test_apply_striation_preserving_filter_1d_lowpass():
    """Test lowpass filtering removes high-frequency noise."""
    np.random.seed(42)

    # Create signal with low-frequency component + high-frequency noise
    rows = np.linspace(0, 10, 100)
    low_freq = np.sin(2 * np.pi * rows / 10)  # Low frequency signal
    high_freq = np.random.randn(100) * 0.5  # High frequency noise

    surface = np.tile((low_freq + high_freq).reshape(-1, 1), (1, 20))

    scan_image = ScanImage(data=surface, scale_x=1e-6, scale_y=1e-6)
    smoothed, mask = apply_striation_preserving_filter_1d(
        scan_image=scan_image,
        cutoff=250e-6,
        is_high_pass=False,
        cut_borders_after_smoothing=False,
    )

    # Smoothed data should have lower variance than input
    assert np.std(smoothed) < np.std(surface)


def test_apply_striation_preserving_filter_1d_highpass():
    """Test highpass filtering removes low-frequency shape."""
    # Create surface with parabolic shape + fine detail
    rows = np.linspace(0, 100, 200)
    shape = 0.01 * (rows - 50) ** 2  # Parabolic shape (range ~25)
    detail = np.sin(2 * np.pi * rows / 5) * 0.1  # Fine detail (range ~0.2)

    surface = np.tile((shape + detail).reshape(-1, 1), (1, 20))

    # Use pixel size that makes cutoff effective for shape removal
    scan_image = ScanImage(data=surface, scale_x=1e-3, scale_y=1e-3)
    residuals, mask = apply_striation_preserving_filter_1d(
        scan_image=scan_image,
        cutoff=50e-3,
        is_high_pass=True,
        cut_borders_after_smoothing=False,
    )

    # Residuals should have much smaller range than input (shape removed)
    assert np.ptp(residuals) < np.ptp(surface) * 0.25


# =============================================================================
# Tests for shape and noise removal
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


def test_shape_noise_removal_filter_sequence():
    """Test filter sequence, border cropping, and mask propagation."""
    np.random.seed(42)

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

    # Test border cropping behavior
    sigma = cutoff_to_gaussian_sigma(2000e-6, scale)
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


def test_shape_noise_removal_short_data():
    """Test that short data is handled without border cropping."""
    np.random.seed(42)

    scale = 1e-6
    sigma = cutoff_to_gaussian_sigma(2000e-6, scale)
    short_height = int(2 * sigma / 0.2) - 5
    width = 150

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


def test_shape_noise_removal_mask_propagation():
    """Test that masks are properly propagated through filtering."""
    np.random.seed(42)

    height, width = 200, 150
    scale = 1e-6

    x = np.arange(height) * scale
    X, _ = np.meshgrid(x, np.arange(width), indexing="ij")

    form = 5e-6 * (X / x.max()) ** 2
    striations = 0.5e-6 * np.sin(2 * np.pi * X / 500e-6)
    noise = 0.1e-6 * np.random.randn(height, width)
    depth_data = form + striations + noise

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


def test_shape_noise_removal_synthetic():
    """Test on synthetic data with known ground truth."""
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


# =============================================================================
# Tests for rotation and alignment
# =============================================================================


def test_rotate_data_by_shifting_profiles():
    """Test rotation by profile shifting."""
    data = np.zeros((50, 50), dtype=float)
    data[25, :] = 1.0

    # 5 degrees = 0.087 radians
    angle_rad = np.radians(5.0)
    rotated = _shear_data_by_shifting_profiles(
        data, angle_rad=angle_rad, cut_y_after_shift=True
    )

    assert rotated.shape[0] < data.shape[0]
    max_positions = np.argmax(rotated, axis=0)
    assert np.std(max_positions) > 0


def test_rotate_image_grad_vector():
    """Test gradient-based striation angle detection."""
    np.random.seed(42)

    height, width = 100, 100
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    angle_rad = np.radians(5.0)
    striations = np.sin(
        2 * np.pi * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)) / 10
    )

    detected_angle = _rotate_image_grad_vector(
        striations,
        scale_x=1e-6,
        subsampling_factor=1,
    )

    assert abs(detected_angle) < 10


def test_fine_align_bullet_marks():
    """Test iterative fine alignment of striated marks."""
    np.random.seed(42)

    height, width = 80, 80
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    angle_input = 2.0
    angle_rad = np.radians(angle_input)
    striations = np.sin(2 * np.pi * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)) / 8)

    scan_image = ScanImage(data=striations, scale_x=1e-6, scale_y=1e-6)
    aligned_scan, _, detected_angle = fine_align_bullet_marks(
        scan_image=scan_image,
        angle_accuracy=0.5,
        cut_y_after_shift=False,
        max_iter=10,
    )

    assert aligned_scan.data.shape[0] > 0
    assert aligned_scan.data.shape[1] > 0
    assert abs(detected_angle) < 45


# =============================================================================
# Tests for profile extraction
# =============================================================================


def test_extract_profile_mean():
    """Test mean profile extraction from 2D data."""
    data = np.zeros((10, 20), dtype=float)
    for i in range(10):
        data[i, :] = i * 2.0

    profile = extract_profile(data, use_mean=True)

    assert profile.shape == (10,)
    assert profile[0] == pytest.approx(0.0)
    assert profile[5] == pytest.approx(10.0)


def test_extract_profile_median():
    """Test median profile extraction from 2D data."""
    data = np.zeros((10, 20), dtype=float)
    for i in range(10):
        data[i, :] = i * 2.0

    profile = extract_profile(data, use_mean=False)

    assert profile.shape == (10,)
    assert profile[5] == pytest.approx(10.0)


def test_extract_profile_with_mask():
    """Test profile extraction with mask."""
    data = np.zeros((10, 20), dtype=float)
    for i in range(10):
        data[i, :] = i * 2.0

    mask = np.ones_like(data, dtype=bool)
    mask[:, 10:] = False

    profile = extract_profile(data, mask=mask, use_mean=True)

    assert profile.shape == (10,)
    assert profile[5] == pytest.approx(10.0)


# =============================================================================
# Tests for full preprocessing pipeline
# =============================================================================


def test_preprocess_data():
    """Test complete preprocess_striations pipeline."""
    np.random.seed(42)

    height, width = 80, 80
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Create synthetic data with form + striations + noise
    form = 0.001 * (Y - height / 2) ** 2
    striations = np.sin(2 * np.pi * X / 10) * 0.01
    noise = np.random.randn(height, width) * 0.001
    depth_data = form + striations + noise

    scan_image = ScanImage(data=depth_data, scale_x=1e-6, scale_y=1e-6)
    params = PreprocessingStriationParams(
        cutoff_hi=2000e-6,
        cutoff_lo=250e-6,
        cut_borders_after_smoothing=False,
        angle_accuracy=0.5,
        max_iter=10,
    )
    aligned, profile, mask, angle = preprocess_data(
        scan_image=scan_image, params=params
    )

    assert aligned.shape[0] > 0
    assert aligned.shape[1] > 0
    assert profile.shape[0] == aligned.shape[0]
    assert abs(angle) < 45


# =============================================================================
# Tests for MarkType and resampling utilities
# =============================================================================


def test_mark_type_scale():
    """Test MarkType.scale property returns correct sampling distance."""
    assert np.isclose(
        MarkType.BULLET_GEA_STRIATION.scale,
        1.5e-6,
        rtol=1e-09,
        atol=1e-09,
    )
    assert np.isclose(
        MarkType.BREECH_FACE_IMPRESSION.scale,
        3.5e-6,
        rtol=1e-09,
        atol=1e-09,
    )
    assert np.isclose(
        MarkType.FIRING_PIN_DRAG_STRIATION.scale,
        1.5e-6,
        rtol=1e-09,
        atol=1e-09,
    )


def test_resample_to_mark_type_scale():
    """Test resampling to target sampling distance."""
    np.random.seed(42)
    data = np.random.randn(100, 100)
    scale_x = 1.0e-6
    scale_y = 1.0e-6
    mark_type = MarkType.BULLET_GEA_STRIATION

    scan_image = ScanImage(data=data, scale_x=scale_x, scale_y=scale_y)
    resampled_scan, _ = resample_scan_image_and_mask(
        scan_image, mask=None, target_scale=mark_type.scale, only_downsample=True
    )

    # With only_downsample=True, data with scale smaller than target (1.0e-6 < 1.5e-6)
    # should be downsampled
    assert resampled_scan.data.shape[0] < data.shape[0]
    assert resampled_scan.data.shape[1] < data.shape[1]
    assert np.isclose(resampled_scan.scale_x, 1.5e-6, rtol=1e-09, atol=1e-09)
    assert np.isclose(resampled_scan.scale_y, 1.5e-6, rtol=1e-09, atol=1e-09)
