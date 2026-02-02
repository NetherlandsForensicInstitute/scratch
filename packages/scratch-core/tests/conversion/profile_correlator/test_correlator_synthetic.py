"""Comprehensive test suite for correlate_profiles with synthetic data.

This module tests the correlate_profiles function using synthetic striation
profiles with known properties to verify alignment accuracy.
"""

import pytest

from conversion.profile_correlator import (
    AlignmentParameters,
    Profile,
)

from .synthetic_helpers import (
    OUTPUT_DIR,
    create_base_profile,
    create_partial_length_profiles,
    create_scaled_profiles,
    create_shifted_profiles,
    run_correlation_with_visualization,
)

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard pixel size for tests (1.5 μm)
PIXEL_SIZE_M = 1.5e-6


class TestIdenticalProfiles:
    """Tests for correlation of identical profiles."""

    def test_identical_profiles_perfect_correlation(self):
        """Identical profiles should have near-perfect correlation."""
        # Create base profile
        base_data = create_base_profile(n_samples=1000, seed=42)

        profile_ref = Profile(depth_data=base_data.copy(), pixel_size=PIXEL_SIZE_M)
        profile_comp = Profile(depth_data=base_data.copy(), pixel_size=PIXEL_SIZE_M)

        params = AlignmentParameters(
            scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
        )

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title="Identical Profiles",
            output_filename="identical_profiles.png",
        )

        # Verify near-perfect correlation
        assert result.correlation_coefficient > 0.999, (
            f"Expected correlation > 0.999, got {result.correlation_coefficient}"
        )

        # Verify high overlap ratio
        assert result.overlap_ratio > 0.99, (
            f"Expected overlap_ratio > 0.99, got {result.overlap_ratio}"
        )


class TestShiftedProfiles:
    """Tests for profiles with translation shifts."""

    @pytest.mark.parametrize(
        "shift_pct,min_corr",
        [
            (3, 0.80),  # Algorithm achieves ~0.88
            (5, 0.80),  # Algorithm achieves ~0.87
            (10, 0.70),  # Algorithm achieves ~0.77
            (20, 0.60),  # Algorithm achieves ~0.65
            (30, 0.90),  # Algorithm achieves ~0.94
            (50, 0.80),  # Algorithm achieves ~0.88
        ],
    )
    def test_shifted_profiles(self, shift_pct: int, min_corr: float):
        """Test alignment of shifted profiles.

        Creates profiles with a known translation shift and verifies the
        correlator can recover the alignment using partial alignment mode.

        :param shift_pct: Shift as percentage of profile length.
        :param min_corr: Minimum expected correlation coefficient.
        """
        n_samples = 1000
        base_data = create_base_profile(n_samples=n_samples, seed=42)

        # Calculate shift in samples from percentage
        shift_samples = int(n_samples * shift_pct / 100)

        profile_ref, profile_comp = create_shifted_profiles(
            base_data,
            shift_samples=shift_samples,
            pixel_size_m=PIXEL_SIZE_M,
        )

        params = AlignmentParameters(
            scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
        )

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title=f"Shifted {shift_pct}%",
            output_filename=f"shifted_{shift_pct:02d}_percent.png",
        )

        # Verify correlation meets minimum
        assert result.correlation_coefficient >= min_corr, (
            f"Expected correlation >= {min_corr}, got {result.correlation_coefficient}"
        )

        # Verify reasonable overlap ratio (larger shifts have lower overlap)
        if shift_pct >= 50:
            min_overlap = 0.3
        elif shift_pct >= 30:
            min_overlap = 0.5
        else:
            min_overlap = 0.6
        assert result.overlap_ratio > min_overlap, (
            f"Expected overlap_ratio > {min_overlap}, got {result.overlap_ratio}"
        )


class TestPartialLengthProfiles:
    """Tests for profiles where comparison is a subset of reference."""

    @pytest.mark.parametrize("length_pct", [50, 30])
    def test_partial_length_profiles(self, length_pct: int):
        """Test partial profile matching where comparison is a subset.

        Creates ref=full, comp=subset and verifies partial matching is triggered.

        :param length_pct: Length of comparison as percentage of reference.
        """
        base_data = create_base_profile(n_samples=1000, seed=42)

        profile_ref, profile_comp = create_partial_length_profiles(
            base_data,
            partial_ratio=length_pct / 100.0,
            pixel_size_m=PIXEL_SIZE_M,
        )

        params = AlignmentParameters(
            scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
        )

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title=f"Partial Length {length_pct}%",
            output_filename=f"partial_length_{length_pct}_percent.png",
        )

        # Should achieve high correlation for matching subset
        assert result.correlation_coefficient > 0.85, (
            f"Expected correlation > 0.85, got {result.correlation_coefficient}"
        )

        # Overlap ratio should be high (relative to shorter profile)
        assert result.overlap_ratio > 0.8, (
            f"Expected overlap_ratio > 0.8, got {result.overlap_ratio}"
        )


class TestScaledProfiles:
    """Tests for profiles with scaling differences."""

    @pytest.mark.parametrize(
        "scale_pct,min_corr",
        [
            (5, 0.95),
            (10, 0.90),
            (20, 0.80),
        ],
    )
    def test_scaled_profiles(self, scale_pct: int, min_corr: float):
        """Test alignment of scaled (stretched) profiles.

        Creates ref=original, comp=stretched and verifies scaling is detected.

        :param scale_pct: Scaling percentage (e.g., 5 means 1.05x stretch).
        :param min_corr: Minimum expected correlation coefficient.
        """
        base_data = create_base_profile(n_samples=1000, seed=42)

        scale_factor = 1.0 + scale_pct / 100.0
        profile_ref, profile_comp = create_scaled_profiles(
            base_data,
            scale_factor=scale_factor,
            pixel_size_m=PIXEL_SIZE_M,
        )

        # Allow scaling detection up to the applied scale
        max_scaling = scale_pct / 100.0 + 0.02  # Add small margin

        params = AlignmentParameters(
            scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
            max_scaling=max_scaling,
        )

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title=f"Scaled {scale_pct}%",
            output_filename=f"scaled_{scale_pct:02d}_percent.png",
        )

        # Verify correlation meets minimum
        assert result.correlation_coefficient >= min_corr, (
            f"Expected correlation >= {min_corr}, got {result.correlation_coefficient}"
        )

        # Verify scale factor is approximately correct
        # The detected scale should be close to 1/scale_factor since we need to
        # "shrink" the comparison to match reference
        expected_detected_scale = 1.0 / scale_factor
        scale_tolerance = 0.05  # 5% tolerance

        assert abs(result.scale_factor - expected_detected_scale) < scale_tolerance, (
            f"Expected scale_factor ~ {expected_detected_scale:.3f}, "
            f"got {result.scale_factor:.3f}"
        )

    def test_custom_threshold(self):
        """Test correlation with different length profiles.

        90% length ratio = 10% length difference. Verifies correlation works.
        """
        base_data = create_base_profile(n_samples=1000, seed=42)

        # 90% length ratio = 10% length difference
        profile_ref, profile_comp = create_partial_length_profiles(
            base_data,
            partial_ratio=0.90,
            pixel_size_m=PIXEL_SIZE_M,
        )

        params = AlignmentParameters(
            scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
        )

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title="Custom Threshold: 10% length diff",
            output_filename="partial_threshold_custom_15.png",
        )

        # Should achieve high correlation for matching subset
        assert result.correlation_coefficient > 0.85, (
            f"Expected correlation > 0.85, got {result.correlation_coefficient}"
        )


class TestFlippedProfiles:
    """Tests with reference and comparison profiles swapped.

    These tests verify that the correlator handles both orderings correctly,
    particularly for partial profiles where the shorter profile could be
    either the reference or the comparison.
    """

    @pytest.mark.parametrize("length_pct", [50, 30])
    def test_partial_length_flipped(self, length_pct: int):
        """Test partial matching with shorter profile as reference.

        Creates ref=subset (shorter), comp=full (longer) - the opposite of
        the normal case. The correlator should handle this by internally
        swapping if needed.

        :param length_pct: Length of reference as percentage of comparison.
        """
        base_data = create_base_profile(n_samples=1000, seed=42)

        # Create profiles but swap them: ref=short, comp=long
        profile_long, profile_short = create_partial_length_profiles(
            base_data,
            partial_ratio=length_pct / 100.0,
            pixel_size_m=PIXEL_SIZE_M,
        )

        # Flip: use short as reference, long as comparison
        profile_ref = profile_short
        profile_comp = profile_long

        params = AlignmentParameters(
            scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
        )

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title=f"Partial Length {length_pct}% (Flipped)",
            output_filename=f"partial_length_{length_pct}_percent_flipped.png",
        )

        # Should achieve high correlation
        assert result.correlation_coefficient > 0.85, (
            f"Expected correlation > 0.85, got {result.correlation_coefficient}"
        )

        # Overlap ratio should be high
        assert result.overlap_ratio > 0.8, (
            f"Expected overlap_ratio > 0.8, got {result.overlap_ratio}"
        )

    @pytest.mark.parametrize(
        "scale_pct,min_corr",
        [
            (5, 0.95),
            (10, 0.90),
        ],
    )
    def test_scaled_flipped(self, scale_pct: int, min_corr: float):
        """Test scaled profiles with reference and comparison swapped.

        Creates ref=stretched, comp=original - the opposite of the normal case.

        :param scale_pct: Scaling percentage.
        :param min_corr: Minimum expected correlation coefficient.
        """
        base_data = create_base_profile(n_samples=1000, seed=42)

        scale_factor = 1.0 + scale_pct / 100.0
        profile_original, profile_stretched = create_scaled_profiles(
            base_data,
            scale_factor=scale_factor,
            pixel_size_m=PIXEL_SIZE_M,
        )

        # Flip: use stretched as reference, original as comparison
        profile_ref = profile_stretched
        profile_comp = profile_original

        max_scaling = scale_pct / 100.0 + 0.02

        params = AlignmentParameters(
            scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
            max_scaling=max_scaling,
        )

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title=f"Scaled {scale_pct}% (Flipped)",
            output_filename=f"scaled_{scale_pct:02d}_percent_flipped.png",
        )

        # Should achieve high correlation
        assert result.correlation_coefficient >= min_corr, (
            f"Expected correlation >= {min_corr}, got {result.correlation_coefficient}"
        )

        # Scale factor should be approximately the applied scale factor
        # (opposite of non-flipped case which expects 1/scale_factor)
        expected_detected_scale = scale_factor
        scale_tolerance = 0.05

        assert abs(result.scale_factor - expected_detected_scale) < scale_tolerance, (
            f"Expected scale_factor ~ {expected_detected_scale:.3f}, "
            f"got {result.scale_factor:.3f}"
        )

    @pytest.mark.parametrize(
        "shift_pct,min_corr",
        [
            (3, 0.75),  # Algorithm achieves ~0.78 (flipped)
            (5, 0.80),  # Algorithm achieves ~0.87
            (10, 0.70),  # Algorithm achieves ~0.77
            (20, 0.60),  # Algorithm achieves ~0.65
            (30, 0.90),  # Algorithm achieves ~0.94
            (50, 0.80),  # Algorithm achieves ~0.88
        ],
    )
    def test_shifted_flipped(self, shift_pct: int, min_corr: float):
        """Test shifted profiles with reference and comparison swapped.

        Uses partial alignment mode since shifted profiles have partial overlap.

        :param shift_pct: Shift as percentage of profile length.
        :param min_corr: Minimum expected correlation coefficient.
        """
        n_samples = 1000
        base_data = create_base_profile(n_samples=n_samples, seed=42)

        # Calculate shift in samples from percentage
        shift_samples = int(n_samples * shift_pct / 100)

        profile_a, profile_b = create_shifted_profiles(
            base_data,
            shift_samples=shift_samples,
            pixel_size_m=PIXEL_SIZE_M,
        )

        # Flip the profiles
        profile_ref = profile_b
        profile_comp = profile_a

        params = AlignmentParameters(
            scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
        )

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title=f"Shifted {shift_pct}% (Flipped)",
            output_filename=f"shifted_{shift_pct:02d}_percent_flipped.png",
        )

        # Verify correlation meets minimum
        assert result.correlation_coefficient >= min_corr, (
            f"Expected correlation >= {min_corr}, got {result.correlation_coefficient}"
        )

        # Verify reasonable overlap ratio (larger shifts have lower overlap)
        if shift_pct >= 50:
            min_overlap = 0.3
        elif shift_pct >= 30:
            min_overlap = 0.5
        else:
            min_overlap = 0.6
        assert result.overlap_ratio > min_overlap, (
            f"Expected overlap_ratio > {min_overlap}, got {result.overlap_ratio}"
        )
