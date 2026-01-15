"""
Tests for fine alignment functions in preprocess_data.py.
"""

import numpy as np
import pytest

from conversion.data_formats import MarkType
from conversion.preprocess_striation.parameters import PreprocessingStriationParams
from conversion.preprocess_striation.preprocess_data import (
    _smooth_2d,
    _rotate_data_by_shifting_profiles,
    _rotate_image_grad_vector,
    fine_align_bullet_marks,
    extract_profile,
    preprocess_data,
)
from conversion.resample import resample_scan_image_and_mask
from container_models.scan_image import ScanImage


def test_smooth_2d():
    """Test 2D Gaussian smoothing with NaN handling."""
    data = np.ones((20, 20), dtype=float)
    data[10, 10] = np.nan

    result = _smooth_2d(data, sigma=2.0)

    assert not np.isnan(result[10, 10])
    assert result[10, 10] == pytest.approx(1.0, rel=0.1)
    assert result[0, 0] == pytest.approx(1.0, rel=0.1)


def test_rotate_data_by_shifting_profiles():
    """Test rotation by profile shifting."""
    data = np.zeros((50, 50), dtype=float)
    data[25, :] = 1.0

    # 5 degrees = 0.087 radians
    angle_rad = np.radians(5.0)
    rotated = _rotate_data_by_shifting_profiles(
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
        extra_sub_samp=1,
    )

    assert abs(detected_angle) < 10


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


def test_resample_mark_type_specific():
    """Test resampling to target sampling distance using resample_scan_image_and_mask."""
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


def test_extract_profile():
    """Test profile extraction from 2D data."""
    data = np.zeros((10, 20), dtype=float)
    for i in range(10):
        data[i, :] = i * 2.0

    profile_mean = extract_profile(data, use_mean=True)
    assert profile_mean.shape == (10,)
    assert profile_mean[0] == pytest.approx(0.0)
    assert profile_mean[5] == pytest.approx(10.0)

    profile_median = extract_profile(data, use_mean=False)
    assert profile_median.shape == (10,)
    assert profile_median[5] == pytest.approx(10.0)

    mask = np.ones_like(data, dtype=bool)
    mask[:, 10:] = False
    profile_masked = extract_profile(data, mask=mask, use_mean=True)
    assert profile_masked.shape == (10,)
    assert profile_masked[5] == pytest.approx(10.0)


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
