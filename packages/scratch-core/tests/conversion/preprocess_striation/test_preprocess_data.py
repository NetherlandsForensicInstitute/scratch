"""
Tests for preprocess_striation.py and related filter functions.
"""

import numpy as np
import pytest
from math import ceil
from scipy.constants import micro

from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType
from ..helper_functions import make_mark
from conversion.filter import (
    apply_striation_preserving_filter_1d,
    cutoff_to_gaussian_sigma,
)
from conversion.filter.gaussian import _apply_nan_weighted_gaussian_1d
from conversion.preprocess_striation import (
    PreprocessingStriationParams,
    apply_shape_noise_removal,
    fine_align_bullet_marks,
    preprocess_striation_mark,
)
from conversion.preprocess_striation.shear import shear_data_by_shifting_profiles
from conversion.preprocess_striation.alignment import _detect_striation_angle


def test_cutoff_to_gaussian_sigma():
    """Test conversion from cutoff wavelength to Gaussian sigma."""
    cutoff = 1000 * micro  # 1000 µm
    pixel_size = 10 * micro  # 10 µm per pixel

    sigma = cutoff_to_gaussian_sigma(cutoff, pixel_size)

    # The function uses: σ = α·λc/√(2π) where α = √(ln(2)/π)
    alpha_gaussian = np.sqrt(np.log(2) / np.pi)
    expected = alpha_gaussian * (cutoff / pixel_size) / np.sqrt(2 * np.pi)
    assert sigma == pytest.approx(expected)


def test_apply_nan_weighted_gaussian_1d():
    """Test that NaN values are handled correctly in 1D Gaussian filter."""
    data = np.ones((20, 10), dtype=float)
    data[10, 5] = np.nan  # Single NaN value

    scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
    result = _apply_nan_weighted_gaussian_1d(scan_image, cutoff_length=10.0, axis=0)

    # NaN positions should be preserved
    assert np.isnan(result[10, 5])
    # Non-NaN regions should remain close to 1.0
    assert result[0, 0] == pytest.approx(1.0, rel=0.1)


def test_apply_striation_preserving_filter_1d_lowpass():
    """Test lowpass filtering removes high-frequency noise."""
    np.random.seed(42)

    # Create signal with low-frequency component + high-frequency noise
    rows = np.linspace(0, 10, 100)
    low_freq = np.sin(2 * np.pi * rows / 10)  # Low frequency signal
    high_freq = np.random.randn(100) * 0.5  # High frequency noise

    surface = np.tile((low_freq + high_freq).reshape(-1, 1), (1, 20))

    scan_image = ScanImage(data=surface, scale_x=micro, scale_y=micro)
    smoothed = apply_striation_preserving_filter_1d(
        scan_image=scan_image,
        cutoff=2.5e-4,
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
    residuals = apply_striation_preserving_filter_1d(
        scan_image=scan_image,
        cutoff=5e-2,
        is_high_pass=True,
        cut_borders_after_smoothing=False,
    )

    # Residuals should have much smaller range than input (shape removed)
    assert np.ptp(residuals) < np.ptp(surface) * 0.25


def test_apply_shape_noise_removal():
    """Test complete form and noise removal pipeline."""
    np.random.seed(42)

    # Create synthetic surface: shape + striations + noise
    rows = np.linspace(0, 100, 300)
    shape = 0.001 * (rows - 50) ** 2  # Parabolic form
    striations = np.sin(2 * np.pi * rows / 10) * 0.01  # Striation pattern
    noise = np.random.randn(300) * 0.001  # Fine noise

    surface = np.tile((shape + striations + noise).reshape(-1, 1), (1, 50))

    scan_image = ScanImage(data=surface, scale_x=micro, scale_y=micro)
    result = apply_shape_noise_removal(
        scan_image=scan_image,
        highpass_cutoff=2e-3,
        lowpass_cutoff=2.5e-4,
    )

    # Result should have form removed (smaller range than input)
    assert np.ptp(result) < np.ptp(surface)


def test_shape_noise_removal_filter_sequence():
    """Test filter sequence and border cropping."""
    np.random.seed(42)

    height, width = 200, 150
    scale = micro

    # Create test data with known components
    x = np.arange(height) * scale
    X, _ = np.meshgrid(x, np.arange(width), indexing="ij")

    form = 5 * micro * (X / x.max()) ** 2  # Large wavelength
    striations = (
        0.5 * micro * np.sin(2 * np.pi * X / (500 * micro))
    )  # Medium wavelength
    noise = 0.1 * micro * np.random.randn(height, width)  # Small wavelength

    depth_data = form + striations + noise

    scan_image = ScanImage(data=depth_data, scale_x=scale, scale_y=scale)
    result = apply_shape_noise_removal(
        scan_image=scan_image,
        highpass_cutoff=2e-3,
        lowpass_cutoff=2.5e-4,
    )

    # Verify form removed (mean near zero)
    assert np.abs(np.mean(result)) < micro, "Form not removed"

    # Verify striations preserved (signal remains)
    assert np.std(result) > 0.05 * micro, "Striations not preserved"

    # Verify noise reduced
    assert np.std(result) < np.std(depth_data), "Noise not reduced"

    # Test border cropping behavior
    sigma = cutoff_to_gaussian_sigma(2000 * micro, scale)
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

    scale = micro
    sigma = cutoff_to_gaussian_sigma(2000 * micro, scale)
    short_height = int(2 * sigma / 0.2) - 5
    width = 150

    short_data = np.random.randn(short_height, width) * micro

    short_scan_image = ScanImage(data=short_data, scale_x=scale, scale_y=scale)
    result_short = apply_shape_noise_removal(
        scan_image=short_scan_image,
        highpass_cutoff=2000 * micro,
    )

    # Short data should not have borders cut (automatic detection)
    assert result_short.shape[0] == short_height, (
        f"Short data borders were cut (got {result_short.shape[0]}, expected {short_height})"
    )


def test_shape_noise_removal_synthetic():
    """Test on synthetic data with known ground truth."""
    np.random.seed(42)

    height, width = 200, 150
    scale = micro

    x = np.arange(height) * scale
    y = np.arange(width) * scale
    X, _ = np.meshgrid(x, y, indexing="ij")

    form = 5 * micro * (X / x.max()) ** 2
    striations = 0.5 * micro * np.sin(2 * np.pi * X / (500 * micro))
    noise = 0.1 * micro * np.random.randn(height, width)

    depth_data = form + striations + noise

    scan_image = ScanImage(data=depth_data, scale_x=scale, scale_y=scale)
    result = apply_shape_noise_removal(
        scan_image=scan_image,
        highpass_cutoff=2000 * micro,
        lowpass_cutoff=250 * micro,
    )

    # Verify form removed (mean near zero)
    assert np.abs(np.mean(result)) < micro, "Form not removed"

    # Verify striations preserved
    std_result = np.std(result)
    assert std_result > 0.1 * micro, "Striations lost"

    # Verify noise reduced (result std should be less than original)
    std_original = np.std(depth_data)
    assert std_result < std_original, "Noise not reduced"


def test_rotate_data_by_shifting_profiles():
    """Test rotation by profile shifting."""
    data = np.zeros((50, 50), dtype=float)
    data[25, :] = 1.0

    # 5 degrees = 0.087 radians
    angle_rad = np.radians(5.0)
    rotated = shear_data_by_shifting_profiles(
        data, angle_rad=angle_rad, cut_y_after_shift=True
    )

    assert rotated.shape[0] < data.shape[0]
    max_positions = np.argmax(rotated, axis=0)
    assert np.std(max_positions) > 0


def test_detect_striation_angle():
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

    scan_image = ScanImage(data=striations, scale_x=micro, scale_y=micro)
    detected_angle = _detect_striation_angle(
        scan_image,
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

    mark = make_mark(
        striations,
        scale_x=micro,
        scale_y=micro,
        mark_type=MarkType.BULLET_GEA_STRIATION,
    )
    aligned_mark, detected_angle = fine_align_bullet_marks(
        mark=mark,
        angle_accuracy=0.5,
        cut_y_after_shift=False,
        max_iter=10,
    )

    assert aligned_mark.scan_image.data.shape[0] > 0
    assert aligned_mark.scan_image.data.shape[1] > 0
    assert abs(detected_angle) < 45


# =============================================================================
# Tests for full preprocessing pipeline
# =============================================================================


def test_preprocess_striation_mark():
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

    input_mark = make_mark(
        depth_data,
        scale_x=micro,
        scale_y=micro,
        mark_type=MarkType.BULLET_LEA_STRIATION,
    )
    params = PreprocessingStriationParams(
        highpass_cutoff=2e-3,
        lowpass_cutoff=2.5e-4,
        cut_borders_after_smoothing=False,
        angle_accuracy=0.5,
        max_iter=10,
    )
    aligned_mark, profile = preprocess_striation_mark(mark=input_mark, params=params)

    aligned = aligned_mark.scan_image.data
    angle = aligned_mark.meta_data.get("total_angle", 0.0)

    assert aligned.shape[0] > 0
    assert aligned.shape[1] > 0
    assert len(profile.heights) == aligned.shape[0]
    assert abs(angle) < 45


# =============================================================================
# Tests for MarkType and resampling utilities
# =============================================================================


def test_mark_type_scale():
    """Test MarkType.scale property returns correct sampling distance."""
    assert np.isclose(
        MarkType.BULLET_GEA_STRIATION.scale,
        1.5 * micro,
        rtol=1e-09,
        atol=1e-09,
    )
    assert np.isclose(
        MarkType.BREECH_FACE_IMPRESSION.scale,
        3.5 * micro,
        rtol=1e-09,
        atol=1e-09,
    )
    assert np.isclose(
        MarkType.FIRING_PIN_DRAG_STRIATION.scale,
        1.5 * micro,
        rtol=1e-09,
        atol=1e-09,
    )
