"""Tests for the transforms module."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from conversion.profile_correlator import (
    Profile,
    equalize_pixel_scale,
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
