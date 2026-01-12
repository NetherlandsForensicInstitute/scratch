"""
Tests for fine alignment functions in preprocess_data.py.
"""

import numpy as np
import pytest

from conversion.preprocessing.preprocess_data import (
    _smooth_2d,
    _remove_zero_image_border,
    _rotate_data_by_shifting_profiles,
    _rotate_image_grad_vector,
    _get_target_sampling_distance,
    _resample_mark_type_specific,
    fine_align_bullet_marks,
    extract_profile,
    preprocess_data,
)


def test_smooth_2d():
    """Test 2D Gaussian smoothing with NaN handling."""
    data = np.ones((20, 20), dtype=float)
    data[10, 10] = np.nan

    result = _smooth_2d(data, sigma=2.0)

    assert not np.isnan(result[10, 10])
    assert result[10, 10] == pytest.approx(1.0, rel=0.1)
    assert result[0, 0] == pytest.approx(1.0, rel=0.1)


def test_remove_zero_image_border():
    """Test that zero borders are correctly removed."""
    data = np.zeros((15, 15), dtype=float)
    mask = np.zeros((15, 15), dtype=bool)

    data[4:10, 3:12] = 1.0
    mask[4:10, 3:12] = True

    cropped_data, cropped_mask = _remove_zero_image_border(data, mask)

    assert cropped_data.shape == (6, 9)
    assert cropped_mask.shape == (6, 9)
    assert cropped_mask.all()


def test_rotate_data_by_shifting_profiles():
    """Test rotation by profile shifting."""
    data = np.zeros((50, 50), dtype=float)
    data[25, :] = 1.0

    rotated = _rotate_data_by_shifting_profiles(data, angle=5.0, cut_y_after_shift=True)

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
        xdim=1e-6,
        extra_sub_samp=1,
    )

    assert abs(detected_angle) < 10


def test_get_target_sampling_distance():
    """Test mark type to sampling distance lookup."""
    assert np.isclose(
        _get_target_sampling_distance("Bullet GEA striation"),
        1.5e-6,
        rtol=1e-09,
        atol=1e-09,
    )
    assert np.isclose(
        _get_target_sampling_distance("Breech face impression"),
        3.5e-6,
        rtol=1e-09,
        atol=1e-09,
    )
    assert np.isclose(
        _get_target_sampling_distance("Firing pin drag striation"),
        1.5e-6,
        rtol=1e-09,
        atol=1e-09,
    )

    with pytest.raises(ValueError, match="not recognized"):
        _get_target_sampling_distance("Unknown mark type")


def test_resample_mark_type_specific():
    """Test resampling to target sampling distance."""
    np.random.seed(42)
    data = np.random.randn(100, 100)
    xdim = 1.0e-6
    ydim = 1.0e-6
    mark_type = "Bullet GEA striation"

    resampled, new_xdim, new_ydim, _ = _resample_mark_type_specific(
        data, xdim, ydim, mark_type
    )

    assert resampled.shape[0] < data.shape[0]
    assert resampled.shape[1] < data.shape[1]
    assert np.isclose(new_xdim, 1.5e-6, rtol=1e-09, atol=1e-09)
    assert np.isclose(new_ydim, 1.5e-6, rtol=1e-09, atol=1e-09)


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

    aligned, _, detected_angle = fine_align_bullet_marks(
        striations,
        xdim=1e-6,
        angle_accuracy=0.5,
        cut_y_after_shift=False,
        max_iter=10,
    )

    assert aligned.shape[0] > 0
    assert aligned.shape[1] > 0
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
    """Test complete preprocessing pipeline."""
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

    aligned, profile, mask, angle = preprocess_data(
        depth_data,
        xdim=1e-6,
        cutoff_hi=2000e-6,
        cutoff_lo=250e-6,
        cut_borders_after_smoothing=False,
        angle_accuracy=0.5,
        max_iter=10,
    )

    assert aligned.shape[0] > 0
    assert aligned.shape[1] > 0
    assert profile.shape[0] == aligned.shape[0]
    assert abs(angle) < 45
