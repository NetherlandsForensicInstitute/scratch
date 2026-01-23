"""Tests for the transforms module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from conversion.profile_correlator import (
    Profile,
    TransformParameters,
    equalize_pixel_scale,
    make_profiles_equal_length,
    apply_transform,
    compute_cumulative_transform,
)


class TestEqualizeSamplingDistance:
    """Tests for equalize_sampling_distance function."""

    def test_same_pixel_size_unchanged(self):
        """Profiles with same pixel size should be returned unchanged."""
        pixel_size = 0.5e-6
        p1 = Profile(np.random.randn(100), pixel_size=pixel_size)
        p2 = Profile(np.random.randn(100), pixel_size=pixel_size)

        p1_out, p2_out = equalize_pixel_scale(p1, p2)

        assert_array_equal(p1_out.depth_data, p1.depth_data)
        assert_array_equal(p2_out.depth_data, p2.depth_data)

    def test_resamples_higher_resolution_profile(self):
        """Higher resolution profile should be resampled to lower resolution."""
        p1 = Profile(np.random.randn(100), pixel_size=1.0e-6)  # Lower resolution
        p2 = Profile(np.random.randn(200), pixel_size=0.5e-6)  # Higher resolution

        p1_out, p2_out = equalize_pixel_scale(p1, p2)

        # p1 should be unchanged (lower resolution)
        assert_array_equal(p1_out.depth_data, p1.depth_data)
        assert p1_out.pixel_size == p1.pixel_size

        # p2 should be resampled to match p1's pixel size
        assert p2_out.pixel_size == p1.pixel_size
        # Length should be approximately halved
        assert len(p2_out.depth_data) == pytest.approx(len(p2.depth_data) / 2, abs=2)

    def test_resolution_limit_cleared_after_resampling(self):
        """Resolution limit should be cleared after resampling."""
        p1 = Profile(np.random.randn(100), pixel_size=1.0e-6, resolution_limit=5e-6)
        p2 = Profile(np.random.randn(200), pixel_size=0.5e-6, resolution_limit=3e-6)

        p1_out, p2_out = equalize_pixel_scale(p1, p2)

        # The resampled profile should have resolution_limit cleared
        assert p2_out.resolution_limit is None
        # The unchanged profile should keep its resolution_limit
        assert p1_out.resolution_limit == 5e-6


class TestMakeProfilesEqualLength:
    """Tests for make_profiles_equal_length function."""

    def test_same_length_unchanged(self):
        """Profiles with same length should be returned unchanged."""
        p1 = Profile(np.random.randn(100), pixel_size=0.5e-6)
        p2 = Profile(np.random.randn(100), pixel_size=0.5e-6)

        p1_out, p2_out = make_profiles_equal_length(p1, p2)

        assert_array_equal(p1_out.depth_data, p1.depth_data)
        assert_array_equal(p2_out.depth_data, p2.depth_data)

    def test_longer_profile_cropped(self):
        """Longer profile should be cropped to match shorter one."""
        p1 = Profile(np.arange(100, dtype=float), pixel_size=0.5e-6)
        p2 = Profile(np.arange(120, dtype=float), pixel_size=0.5e-6)

        p1_out, p2_out = make_profiles_equal_length(p1, p2)

        assert isinstance(p1_out, Profile)
        assert isinstance(p2_out, Profile)
        assert p1_out.length == 100
        assert p2_out.length == 100

    def test_symmetric_cropping(self):
        """Cropping should be symmetric (equal from both ends)."""
        p1 = Profile(np.arange(100, dtype=float), pixel_size=0.5e-6)
        p2 = Profile(np.arange(120, dtype=float), pixel_size=0.5e-6)

        p1_out, p2_out = make_profiles_equal_length(p1, p2)

        assert isinstance(p2_out, Profile)
        # p2 should be cropped by 10 from each end (diff=20, split equally)
        assert p2_out.depth_data[0] == 10  # Start at index 10
        assert p2_out.depth_data[-1] == 109  # End at index 109

    def test_works_with_profile_objects(self):
        """Should work with Profile objects."""
        p1 = Profile(np.arange(100, dtype=float), pixel_size=0.5e-6)
        p2 = Profile(np.arange(120, dtype=float), pixel_size=0.5e-6)

        p1_out, p2_out = make_profiles_equal_length(p1, p2)

        assert isinstance(p1_out, Profile)
        assert isinstance(p2_out, Profile)
        assert p1_out.length == 100
        assert p2_out.length == 100


class TestApplyTransform:
    """Tests for apply_transform function."""

    def test_zero_translation_no_change(self):
        """Zero translation and unit scaling should not change profile."""
        data = np.random.randn(100)
        profile = Profile(data, pixel_size=0.5e-6)
        transform = TransformParameters(translation=0.0, scaling=1.0)

        result = apply_transform(profile, transform)

        assert_allclose(result, data, atol=1e-10)

    def test_positive_translation_shifts_right(self):
        """Positive translation should shift the profile right (data appears later).

        The transform model is: xx_trans = xx * scaling + translation
        This means data at original position i appears at position i + translation
        in the output.
        """
        data = np.zeros(20)
        data[10] = 1.0  # Spike at position 10
        profile = Profile(data, pixel_size=0.5e-6)
        transform = TransformParameters(translation=5.0, scaling=1.0)

        result = apply_transform(profile, transform)

        # The spike should move to around position 15 (10 + 5)
        peak_pos = np.argmax(result)
        assert peak_pos == pytest.approx(15, abs=1)

    def test_negative_translation_shifts_left(self):
        """Negative translation should shift the profile left (data appears earlier)."""
        data = np.zeros(20)
        data[10] = 1.0  # Spike at position 10
        profile = Profile(data, pixel_size=0.5e-6)
        transform = TransformParameters(translation=-3.0, scaling=1.0)

        result = apply_transform(profile, transform)

        # The spike should move to around position 7 (10 - 3)
        peak_pos = np.argmax(result)
        assert peak_pos == pytest.approx(7, abs=1)

    def test_scaling_stretches_profile(self):
        """Scaling > 1 should stretch the profile (spread data over more positions).

        With scaling > 1, the same features appear spread over a wider range.
        The xx_trans coordinates are spread out, so interpolation spreads the data.
        """
        # Create a simple triangle peak
        data = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 0, 10)])
        profile = Profile(data, pixel_size=0.5e-6)
        transform = TransformParameters(translation=0.0, scaling=1.1)

        result = apply_transform(profile, transform)

        # After stretching, the peak should be wider in sample coordinates
        # because the data is spread out over more positions
        orig_peak_width = np.sum(data > 0.5)
        result_peak_width = np.sum(result > 0.5)
        assert result_peak_width >= orig_peak_width

    def test_multiple_transforms_composed(self):
        """Multiple transforms should be composed correctly."""
        data = np.zeros(100)
        data[50] = 1.0
        profile = Profile(data, pixel_size=0.5e-6)

        transforms = [
            TransformParameters(translation=10.0, scaling=1.0),  # Move by 10
            TransformParameters(translation=5.0, scaling=1.0),  # Move by 5 more
        ]

        result = apply_transform(profile, transforms)

        # Total translation should be 15, so spike moves from 50 to 65
        peak_pos = np.argmax(result)
        assert peak_pos == pytest.approx(65, abs=1)  # 50 + 15 = 65

    def test_fill_value_for_extrapolation(self):
        """Positions outside bounds should be filled with fill_value."""
        data = np.ones(20)
        profile = Profile(data, pixel_size=0.5e-6)
        transform = TransformParameters(translation=10.0, scaling=1.0)

        result = apply_transform(profile, transform, fill_value=0.0)

        # First 10 positions should be filled with 0
        assert_allclose(result[:10], 0.0, atol=1e-10)


class TestComputeCumulativeTransform:
    """Tests for compute_cumulative_transform function."""

    def test_single_transform(self):
        """Single transform should return its values directly."""
        transforms = [TransformParameters(translation=5.0, scaling=1.02)]

        trans, scale = compute_cumulative_transform(transforms)

        assert trans == 5.0
        assert scale == 1.02

    def test_empty_transforms(self):
        """Empty transforms should return identity (0, 1)."""
        trans, scale = compute_cumulative_transform([])

        assert trans == 0.0
        assert scale == 1.0

    def test_multiple_transforms(self):
        """Multiple transforms should be composed correctly."""
        transforms = [
            TransformParameters(translation=5.0, scaling=1.0),
            TransformParameters(translation=3.0, scaling=1.1),
        ]

        trans, scale = compute_cumulative_transform(transforms)

        # Composition: t_total = s2 * t1 + t2 = 1.1 * 5.0 + 3.0 = 8.5
        # s_total = s1 * s2 = 1.0 * 1.1 = 1.1
        assert_allclose(trans, 8.5, atol=1e-10)
        assert_allclose(scale, 1.1, atol=1e-10)

    def test_three_transforms(self):
        """Three transforms should compose correctly."""
        transforms = [
            TransformParameters(translation=2.0, scaling=1.0),
            TransformParameters(translation=3.0, scaling=1.1),
            TransformParameters(translation=1.0, scaling=1.0),
        ]

        trans, scale = compute_cumulative_transform(transforms)

        # After first two: t = 1.1 * 2 + 3 = 5.2, s = 1.1
        # After third: t = 1.0 * 5.2 + 1 = 6.2, s = 1.1
        assert_allclose(trans, 6.2, atol=1e-10)
        assert_allclose(scale, 1.1, atol=1e-10)
