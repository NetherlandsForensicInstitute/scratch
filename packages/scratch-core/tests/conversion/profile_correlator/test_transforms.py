"""Tests for the transforms module."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.constants import micro

from conversion.profile_correlator import (
    Profile,
    equalize_pixel_scale,
)


class TestEqualizePixelScale:
    """Tests for equalize_pixel_scale function."""

    def test_same_pixel_size_unchanged(self):
        """Profiles with same pixel size should be returned unchanged."""
        pixel_size = 0.5 * micro
        p1 = Profile(np.random.randn(100), pixel_size=pixel_size)
        p2 = Profile(np.random.randn(100), pixel_size=pixel_size)

        p1_out, p2_out = equalize_pixel_scale(p1, p2)

        assert_array_equal(p1_out.heights, p1.heights)
        assert_array_equal(p2_out.heights, p2.heights)

    def test_downsamples_higher_resolution_profile(self):
        """Higher resolution profile should be downsampled to lower resolution."""
        p1 = Profile(np.random.randn(100), pixel_size=micro)  # Lower resolution
        p2 = Profile(np.random.randn(200), pixel_size=0.5 * micro)  # Higher resolution

        p1_out, p2_out = equalize_pixel_scale(p1, p2)

        # p1 should be unchanged (lower resolution)
        assert_array_equal(p1_out.heights, p1.heights)
        assert p1_out.pixel_size == p1.pixel_size

        # p2 should be downsampled to match p1's pixel size
        assert p2_out.pixel_size == p1.pixel_size
        # Length should be approximately halved
        assert len(p2_out.heights) == pytest.approx(len(p2.heights) / 2, abs=2)
