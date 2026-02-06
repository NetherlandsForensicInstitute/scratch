"""Tests for the correlator module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d

from conversion.profile_correlator import (
    AlignmentParameters,
    StriationComparisonResults,
    Profile,
    correlate_profiles,
)
from .conftest import make_synthetic_striation_profile, make_shifted_profile

PIXEL_SIZE_M = 1.5e-6  # 1.5 μm


# --- Synthetic profile helpers ---


def create_base_profile(n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate a striation-like profile with multiple sine frequencies."""
    np.random.seed(seed)
    x = np.linspace(0, 20 * np.pi, n_samples)
    data = np.sin(x) * 0.5e-6
    data += np.sin(2.3 * x) * 0.2e-6
    data += np.sin(0.7 * x) * 0.15e-6
    data += np.random.normal(0, 0.01e-6, n_samples)
    return data


def create_shifted_profiles(
    base: np.ndarray, shift_samples: int
) -> tuple[Profile, Profile]:
    """Create two profiles with a known shift."""
    n = len(base)
    extended_length = n + shift_samples
    x = np.linspace(0, 20 * np.pi * extended_length / n, extended_length)
    extended = np.sin(x) * 0.5e-6 + np.sin(2.3 * x) * 0.2e-6 + np.sin(0.7 * x) * 0.15e-6
    ref_data = extended[:n].copy()
    comp_data = extended[shift_samples : shift_samples + n].copy()
    return (
        Profile(heights=ref_data, pixel_size=PIXEL_SIZE_M),
        Profile(heights=comp_data, pixel_size=PIXEL_SIZE_M),
    )


def create_partial_profiles(
    base: np.ndarray, partial_ratio: float
) -> tuple[Profile, Profile]:
    """Create profiles where comparison is a subset of reference."""
    n = len(base)
    partial_len = int(n * partial_ratio)
    start = (n - partial_len) // 2
    return (
        Profile(heights=base.copy(), pixel_size=PIXEL_SIZE_M),
        Profile(
            heights=base[start : start + partial_len].copy(), pixel_size=PIXEL_SIZE_M
        ),
    )


def create_scaled_profiles(
    base: np.ndarray, scale_factor: float
) -> tuple[Profile, Profile]:
    """Create profiles where comparison is stretched."""
    n = len(base)
    x_orig = np.arange(n)
    interp = interp1d(x_orig, base, kind="cubic", fill_value="extrapolate")  # type: ignore[arg-type]
    x_scaled = np.arange(n) / scale_factor
    comp_data = interp(x_scaled)
    return (
        Profile(heights=base.copy(), pixel_size=PIXEL_SIZE_M),
        Profile(heights=comp_data, pixel_size=PIXEL_SIZE_M),
    )


# --- Basic functionality tests ---


class TestCorrelateProfilesBasic:
    """Basic functionality tests for correlate_profiles."""

    def test_returns_comparison_results(self):
        """Should return a StriationComparisonResults object."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        comp = make_shifted_profile(ref, 10.0, seed=43)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert isinstance(result, StriationComparisonResults)

    def test_correlation_coefficient_valid(self):
        """Correlation coefficient should be computed and in valid range."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        comp = make_shifted_profile(ref, 10.0, seed=43)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert not np.isnan(result.correlation_coefficient)
        assert -1 <= result.correlation_coefficient <= 1

    def test_position_shift_computed(self):
        """Position shift should be computed."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        comp = make_shifted_profile(ref, 10.0, seed=43)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert not np.isnan(result.position_shift)

    def test_scale_factor_computed(self):
        """Scale factor should be computed."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        comp = make_shifted_profile(ref, 0.0, 1.01, seed=43)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert not np.isnan(result.scale_factor)

    def test_roughness_metrics_computed(self):
        """Roughness metrics (Sa, Sq) should be computed."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        comp = make_shifted_profile(ref, 10.0, seed=43)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert not np.isnan(result.sa_ref)
        assert not np.isnan(result.mean_square_ref)
        assert result.sa_ref > 0
        assert result.mean_square_ref > 0

    def test_overlap_metrics_computed(self):
        """Overlap length and ratio should be computed."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        comp = make_shifted_profile(ref, 10.0, seed=43)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert not np.isnan(result.overlap_length)
        assert result.overlap_length > 0

    def test_pixel_sizes_recorded(self):
        """Pixel sizes should be recorded in results."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        comp = make_synthetic_striation_profile(n_samples=1000, seed=43)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert_allclose(result.pixel_size, ref.pixel_size, atol=1e-16)

    def test_default_parameters_used(self):
        """Should work with default parameters when none provided."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        comp = make_shifted_profile(ref, 10.0, seed=43)
        result = correlate_profiles(ref, comp)
        assert result is not None
        assert not np.isnan(result.correlation_coefficient)


# --- Synthetic profile alignment tests ---


class TestIdenticalProfiles:
    """Tests for identical profiles."""

    def test_identical_profiles_perfect_correlation(self):
        """Identical profiles should have near-perfect correlation."""
        base = create_base_profile(n_samples=1000, seed=42)
        ref = Profile(heights=base.copy(), pixel_size=PIXEL_SIZE_M)
        comp = Profile(heights=base.copy(), pixel_size=PIXEL_SIZE_M)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert result.correlation_coefficient > 0.999
        assert result.overlap_ratio > 0.99


class TestShiftedProfiles:
    """Tests for profiles with translation shifts."""

    @pytest.mark.parametrize(
        "shift_pct,min_corr",
        [(3, 0.80), (5, 0.80), (10, 0.70), (20, 0.60), (30, 0.90), (50, 0.80)],
    )
    def test_shifted_profiles(self, shift_pct: int, min_corr: float):
        """Shifted profiles should align with good correlation."""
        base = create_base_profile(n_samples=1000, seed=42)
        ref, comp = create_shifted_profiles(base, int(1000 * shift_pct / 100))
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert result.correlation_coefficient >= min_corr

    @pytest.mark.parametrize(
        "shift_pct,min_corr",
        [(3, 0.75), (5, 0.80), (10, 0.70), (20, 0.60), (30, 0.90), (50, 0.80)],
    )
    def test_shifted_profiles_flipped(self, shift_pct: int, min_corr: float):
        """Shifted profiles work with swapped order."""
        base = create_base_profile(n_samples=1000, seed=42)
        a, b = create_shifted_profiles(base, int(1000 * shift_pct / 100))
        result = correlate_profiles(b, a, AlignmentParameters())
        assert result is not None
        assert result.correlation_coefficient >= min_corr


class TestPartialProfiles:
    """Tests for partial profile matching."""

    @pytest.mark.parametrize(
        "length_pct, expected_overlap",
        [
            (50, 3 / 4),
            (30, 0.65),
        ],
    )
    def test_partial_profiles(self, length_pct: int, expected_overlap: float):
        """Partial profiles should match with high correlation."""
        base = create_base_profile(n_samples=1000, seed=42)
        ref, comp = create_partial_profiles(base, length_pct / 100.0)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None
        assert result.correlation_coefficient == 1
        assert result.overlap_ratio == pytest.approx(expected_overlap, rel=1e-6)

    @pytest.mark.parametrize(
        "length_pct, expected_overlap",
        [
            (50, 3 / 4),
            (30, 0.65),
        ],
    )
    def test_partial_profiles_flipped(self, length_pct: int, expected_overlap: float):
        """Partial matching works with shorter profile as reference."""
        base = create_base_profile(n_samples=1000, seed=42)
        long, short = create_partial_profiles(base, length_pct / 100.0)
        result = correlate_profiles(short, long, AlignmentParameters())
        assert result is not None
        assert result.correlation_coefficient == 1
        assert result.overlap_ratio == pytest.approx(expected_overlap, rel=1e-6)


class TestScaledProfiles:
    """Tests for profiles with scaling differences."""

    @pytest.mark.parametrize("scale_pct", [5, 10, 20])
    def test_scaled_profiles(self, scale_pct: int):
        """Scaled profiles should be detected and aligned."""
        base = create_base_profile(n_samples=1000, seed=42)
        scale = 1.0 + scale_pct / 100.0
        ref, comp = create_scaled_profiles(base, scale)
        params = AlignmentParameters(max_scaling=scale_pct / 100.0)
        result = correlate_profiles(ref, comp, params)
        assert result is not None
        assert result.correlation_coefficient >= 0.999
        assert abs(result.scale_factor - scale) == 0

    @pytest.mark.parametrize("scale_pct", [5, 10])
    def test_scaled_profiles_flipped(self, scale_pct: int):
        """Scaled profiles work with stretched as reference."""
        base = create_base_profile(n_samples=1000, seed=42)
        scale = 1.0 + scale_pct / 100.0
        original, stretched = create_scaled_profiles(base, scale)
        params = AlignmentParameters(max_scaling=scale_pct / 100.0)
        result = correlate_profiles(stretched, original, params)
        assert result is not None
        assert result.correlation_coefficient >= 0.999
        assert abs(result.scale_factor - 1 / scale) == 0


# --- Edge case tests ---


class TestEdgeCases:
    """Edge case tests for correlate_profiles."""

    def test_constant_profile(self):
        """Constant profiles return None (no valid correlation possible)."""
        ref = Profile(np.ones(500), pixel_size=0.5e-6)
        comp = Profile(np.ones(500) * 2, pixel_size=0.5e-6)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is None

    def test_very_short_profiles(self):
        """Very short profiles return None when below min_overlap_distance."""
        ref = Profile(np.random.randn(50), pixel_size=0.5e-6)
        comp = Profile(np.random.randn(50), pixel_size=0.5e-6)
        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is None

    def test_different_length_profiles(self):
        """Profiles with different lengths should correlate."""
        ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        partial = Profile(
            heights=ref.heights[100:900].copy(),
            pixel_size=ref.pixel_size,
        )
        result = correlate_profiles(ref, partial, AlignmentParameters())
        assert result is not None
        assert not np.isnan(result.correlation_coefficient)
