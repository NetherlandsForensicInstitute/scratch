"""Tests for the mark_correlator module."""

import numpy as np
import pytest

from conversion.profile_correlator import (
    AlignmentParameters,
    MarkCorrelationResult,
    Profile,
    correlate_profiles,
    correlate_striation_marks,
)
from .conftest import (
    striation_mark,
    make_synthetic_striation_profile,
    make_shifted_profile,
)

PIXEL_SIZE_M = 1.5e-6  # 1.5 μm
N_COLS = 50


@pytest.mark.integration
class TestCorrelateStriationMarksBasic:
    """Basic functionality tests for correlate_striation_marks."""

    def test_returns_mark_correlation_result(self):
        """Should return a MarkCorrelationResult for valid inputs."""
        profile_reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        profile_compared = make_shifted_profile(profile_reference, 10.0, seed=43)
        mark_reference = striation_mark(profile_reference)
        mark_compared = striation_mark(profile_compared)

        result = correlate_striation_marks(
            mark_reference, mark_compared, profile_reference, profile_compared
        )

        assert isinstance(result, MarkCorrelationResult)

    def test_aligned_marks_have_equal_row_count(self):
        """Both aligned 2D marks should have the same number of rows."""
        profile_reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        profile_compared = make_shifted_profile(profile_reference, 10.0, seed=43)
        mark_reference = striation_mark(profile_reference)
        mark_compared = striation_mark(profile_compared)

        result = correlate_striation_marks(
            mark_reference, mark_compared, profile_reference, profile_compared
        )

        assert result is not None
        n_rows_reference = result.mark_reference_aligned.scan_image.height
        n_rows_compared = result.mark_compared_aligned.scan_image.height
        assert n_rows_reference == n_rows_compared

    def test_aligned_profiles_have_equal_length(self):
        """Both aligned profiles should have the same number of samples."""
        profile_reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        profile_compared = make_shifted_profile(profile_reference, 10.0, seed=43)
        mark_reference = striation_mark(profile_reference)
        mark_compared = striation_mark(profile_compared)

        result = correlate_striation_marks(
            mark_reference, mark_compared, profile_reference, profile_compared
        )

        assert result is not None
        assert len(result.profile_reference_aligned.heights) == len(
            result.profile_compared_aligned.heights
        )

    def test_aligned_mark_rows_match_profile_length(self):
        """Aligned mark row count should equal the aligned profile length."""
        profile_reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        profile_compared = make_shifted_profile(profile_reference, 10.0, seed=43)
        mark_reference = striation_mark(profile_reference)
        mark_compared = striation_mark(profile_compared)

        result = correlate_striation_marks(
            mark_reference, mark_compared, profile_reference, profile_compared
        )

        assert result is not None
        assert result.mark_reference_aligned.scan_image.height == len(
            result.profile_reference_aligned.heights
        )
        assert result.mark_compared_aligned.scan_image.height == len(
            result.profile_compared_aligned.heights
        )

    def test_column_count_preserved(self):
        """Aligned marks should retain the original number of columns."""
        profile_reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        profile_compared = make_shifted_profile(profile_reference, 10.0, seed=43)
        mark_reference = striation_mark(profile_reference)
        mark_compared = striation_mark(profile_compared)

        result = correlate_striation_marks(
            mark_reference, mark_compared, profile_reference, profile_compared
        )

        assert result is not None
        assert result.mark_reference_aligned.scan_image.width == N_COLS
        assert result.mark_compared_aligned.scan_image.width == N_COLS

    def test_comparison_results_consistent_with_correlate_profiles(self):
        """comparison_results should equal what correlate_profiles returns directly."""
        profile_reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        profile_compared = make_shifted_profile(profile_reference, 10.0, seed=43)
        mark_reference = striation_mark(profile_reference)
        mark_compared = striation_mark(profile_compared)
        params = AlignmentParameters()

        result = correlate_striation_marks(
            mark_reference, mark_compared, profile_reference, profile_compared, params
        )
        direct = correlate_profiles(profile_reference, profile_compared, params)

        assert result is not None
        assert direct is not None
        assert result.comparison_results == direct


@pytest.mark.integration
class TestCorrelateStriationMarksEdgeCases:
    """Edge case tests for correlate_striation_marks."""

    def test_returns_none_for_short_profiles(self):
        """Too-short profiles should return None."""
        profile_reference = Profile(np.random.randn(50), pixel_size=PIXEL_SIZE_M)
        profile_compared = Profile(np.random.randn(50), pixel_size=PIXEL_SIZE_M)
        mark_reference = striation_mark(profile_reference)
        mark_compared = striation_mark(profile_compared)

        result = correlate_striation_marks(
            mark_reference, mark_compared, profile_reference, profile_compared
        )

        assert result is None

    def test_aligned_overlap_smaller_than_input(self):
        """Aligned marks should be a strict subset of the input mark rows."""
        profile_reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        profile_compared = make_shifted_profile(profile_reference, 50.0, seed=43)
        mark_reference = striation_mark(profile_reference)
        mark_compared = striation_mark(profile_compared)

        result = correlate_striation_marks(
            mark_reference, mark_compared, profile_reference, profile_compared
        )

        assert result is not None
        assert (
            result.mark_reference_aligned.scan_image.height
            <= mark_reference.scan_image.height
        )
