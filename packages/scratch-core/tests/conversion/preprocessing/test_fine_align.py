"""
Tests for fine_align.py.
One test per function.
"""

import numpy as np
import pytest

from conversion.preprocessing.fine_align import (
    smooth_2d,
    remove_zero_image_border,
    rotate_data_by_shifting_profiles,
    rotate_image_grad_vector,
    get_target_sampling_distance,
    resample_mark_type_specific,
    fine_align_bullet_marks,
    extract_profile,
)


def test_smooth_2d():
    """Test 2D Gaussian smoothing with NaN handling."""
    data = np.ones((20, 20), dtype=float)
    data[10, 10] = np.nan  # Single NaN value

    result = smooth_2d(data, sigma=2.0)

    # NaN should be interpolated from neighbors
    assert not np.isnan(result[10, 10])
    # Result at NaN location should be close to 1.0 (surrounded by 1s)
    assert result[10, 10] == pytest.approx(1.0, rel=0.1)
    # Corners should remain close to 1.0
    assert result[0, 0] == pytest.approx(1.0, rel=0.1)


def test_remove_zero_image_border():
    """Test that zero borders are correctly removed."""
    data = np.zeros((15, 15), dtype=float)
    mask = np.zeros((15, 15), dtype=bool)

    # Valid data only in center region
    data[4:10, 3:12] = 1.0
    mask[4:10, 3:12] = True

    cropped_data, cropped_mask = remove_zero_image_border(data, mask)

    assert cropped_data.shape == (6, 9)
    assert cropped_mask.shape == (6, 9)
    assert cropped_mask.all()  # All cropped region should be valid


def test_rotate_data_by_shifting_profiles():
    """Test rotation by profile shifting."""
    # Create simple data with horizontal line
    data = np.zeros((50, 50), dtype=float)
    data[25, :] = 1.0  # Horizontal line in middle

    # Rotate by small angle
    rotated = rotate_data_by_shifting_profiles(data, angle=5.0, cut_y_after_shift=True)

    # Output should be smaller due to border cropping
    assert rotated.shape[0] < data.shape[0]
    # The line should now be diagonal (shifted)
    # Check that max values are not all in same row
    max_positions = np.argmax(rotated, axis=0)
    assert np.std(max_positions) > 0  # Positions should vary across columns


def test_rotate_image_grad_vector():
    """Test gradient-based striation angle detection."""
    np.random.seed(42)

    # Create data with striations at a known angle
    height, width = 100, 100
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Create diagonal striations (approximately 5 degrees)
    angle_rad = np.radians(5.0)
    striations = np.sin(
        2 * np.pi * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)) / 10
    )

    detected_angle = rotate_image_grad_vector(
        striations,
        xdim=1e-6,
        extra_sub_samp=1,
    )

    # Detected angle should be close to the input angle
    # Allow some tolerance due to discrete sampling
    assert abs(detected_angle) < 10  # Should detect an angle in valid range


def test_get_target_sampling_distance():
    """Test mark type to sampling distance lookup."""
    # Test known mark types
    assert get_target_sampling_distance("Bullet GEA striation") == 1.5e-6
    assert get_target_sampling_distance("Breech face impression") == 3.5e-6
    assert get_target_sampling_distance("Firing pin drag striation") == 1.5e-6

    # Test unknown mark type raises error
    with pytest.raises(ValueError, match="not recognized"):
        get_target_sampling_distance("Unknown mark type")


def test_resample_mark_type_specific():
    """Test resampling to target sampling distance."""
    np.random.seed(42)
    data = np.random.randn(100, 100)
    xdim = 1.0e-6  # 1 µm
    ydim = 1.0e-6
    mark_type = "Bullet GEA striation"  # Target: 1.5 µm

    resampled, new_xdim, new_ydim, _ = resample_mark_type_specific(
        data, xdim, ydim, mark_type
    )

    # Data should be downsampled (1.0 µm -> 1.5 µm means smaller output)
    assert resampled.shape[0] < data.shape[0]
    assert resampled.shape[1] < data.shape[1]
    assert new_xdim == 1.5e-6
    assert new_ydim == 1.5e-6


def test_fine_align_bullet_marks():
    """Test iterative fine alignment of striated marks."""
    np.random.seed(42)

    # Create synthetic striated surface with slight rotation
    height, width = 80, 80
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Striations at small angle
    angle_input = 2.0  # degrees
    angle_rad = np.radians(angle_input)
    striations = np.sin(2 * np.pi * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)) / 8)

    aligned, mask, detected_angle = fine_align_bullet_marks(
        striations,
        xdim=1e-6,
        angle_accuracy=0.5,
        cut_y_after_shift=False,
        max_iter=10,
    )

    # Should return valid data
    assert aligned.shape[0] > 0
    assert aligned.shape[1] > 0
    # Detected angle should be reasonable (not failed)
    assert abs(detected_angle) < 45


def test_extract_profile():
    """Test profile extraction from 2D data."""
    # Create data with known row means
    data = np.zeros((10, 20), dtype=float)
    for i in range(10):
        data[i, :] = i * 2.0  # Each row has constant value

    # Test mean profile
    profile_mean = extract_profile(data, use_mean=True)
    assert profile_mean.shape == (10,)
    assert profile_mean[0] == pytest.approx(0.0)
    assert profile_mean[5] == pytest.approx(10.0)

    # Test median profile
    profile_median = extract_profile(data, use_mean=False)
    assert profile_median.shape == (10,)
    assert profile_median[5] == pytest.approx(10.0)

    # Test with mask
    mask = np.ones_like(data, dtype=bool)
    mask[:, 10:] = False  # Mask out right half
    profile_masked = extract_profile(data, mask=mask, use_mean=True)
    assert profile_masked.shape == (10,)
    # Should still get same values since all columns have same value per row
    assert profile_masked[5] == pytest.approx(10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
