"""Tests for the correlator module (main entry point)."""

import numpy as np
from numpy.testing import assert_allclose

from conversion.profile_correlator import (
    Profile,
    AlignmentParameters,
    ComparisonResults,
    correlate_profiles,
)
from .conftest import (
    make_synthetic_striation_profile,
    make_shifted_profile,
)


class TestCorrelateProfiles:
    """Tests for correlate_profiles function."""

    def test_returns_comparison_results(self):
        """Should return a ComparisonResults object."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert isinstance(result, ComparisonResults)

    def test_correlation_coefficient_populated(self):
        """Correlation coefficient should be computed and valid."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert not np.isnan(result.correlation_coefficient)
        assert -1 <= result.correlation_coefficient <= 1

    def test_identical_profiles_high_correlation(self):
        """Identical profiles should have very high correlation."""
        profile = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_copy = Profile(
            depth_data=profile.depth_data.copy(),
            pixel_size=profile.pixel_size,
        )

        params = AlignmentParameters()

        result = correlate_profiles(profile, profile_copy, params)

        assert result.correlation_coefficient > 0.99

    def test_shifted_profiles_aligned(self):
        """Shifted profiles should be aligned with reasonable correlation."""
        np.random.seed(42)
        n_samples = 500
        pixel_size = 0.5e-6

        # Create a clean sine wave
        x = np.linspace(0, 4 * np.pi, n_samples)
        ref_data = np.sin(x) * 1e-6

        # Create shifted version
        shift_samples = 5
        comp_data = np.roll(ref_data, shift_samples)
        comp_data[:shift_samples] = ref_data[:shift_samples]

        profile_ref = Profile(depth_data=ref_data, pixel_size=pixel_size)
        profile_comp = Profile(depth_data=comp_data, pixel_size=pixel_size)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert result.correlation_coefficient > 0.8

    def test_position_shift_computed(self):
        """Position shift should be computed in micrometers."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert not np.isnan(result.position_shift)

    def test_scale_factor_computed(self):
        """Scale factor should be computed."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 0.0, 1.01, seed=43)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert not np.isnan(result.scale_factor)

    def test_roughness_metrics_computed(self):
        """Roughness metrics (Sa, Sq) should be computed."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert not np.isnan(result.sa_ref)
        assert not np.isnan(result.sq_ref)
        assert not np.isnan(result.sa_comp)
        assert not np.isnan(result.sq_comp)
        assert result.sa_ref > 0
        assert result.sq_ref > 0

    def test_signature_differences_computed(self):
        """Signature difference metrics should be computed."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert not np.isnan(result.ds_ref_norm)
        assert not np.isnan(result.ds_comp_norm)
        assert not np.isnan(result.ds_combined)

    def test_overlap_metrics_computed(self):
        """Overlap length and ratio should be computed."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert not np.isnan(result.overlap_length)
        assert result.overlap_length > 0

    def test_pixel_sizes_recorded(self):
        """Pixel sizes should be recorded in results."""
        pixel_size = 0.5e-6
        profile_ref = Profile(np.random.randn(500), pixel_size=pixel_size)
        profile_comp = Profile(np.random.randn(500), pixel_size=pixel_size)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert_allclose(result.pixel_size_ref, 0.5e-6, atol=1e-16)
        assert_allclose(result.pixel_size_comp, 0.5e-6, atol=1e-16)

    def test_equalizes_different_pixel_sizes(self):
        """Profiles with different pixel sizes should be equalized."""
        profile_ref = Profile(np.random.randn(500), pixel_size=1.0e-6)
        profile_comp = Profile(np.random.randn(1000), pixel_size=0.5e-6)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        # After equalization, both should have same pixel size
        assert_allclose(result.pixel_size_ref, result.pixel_size_comp, atol=1e-10)

    def test_different_length_profiles(self):
        """Profiles with different lengths should still correlate."""
        # Create reference and shorter profile
        # With pixel_size=0.5e-6 m, need at least 400 samples for 200 μm minimum overlap
        profile_ref = make_synthetic_striation_profile(n_samples=1000, seed=42)
        partial_data = profile_ref.depth_data[200:700].copy()  # 500 samples = 250 μm
        profile_partial = Profile(
            depth_data=partial_data, pixel_size=profile_ref.pixel_size
        )

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_partial, params)

        # Should get a valid correlation
        assert not np.isnan(result.correlation_coefficient)

    def test_same_length_profiles(self):
        """Profiles with same lengths should correlate."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_synthetic_striation_profile(n_samples=500, seed=43)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        # Should get a valid correlation
        assert not np.isnan(result.correlation_coefficient)

    def test_is_profile_comparison_flag(self):
        """is_profile_comparison flag should be True."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        result = correlate_profiles(profile_ref, profile_comp)

        assert result.is_profile_comparison is True

    def test_default_parameters_used(self):
        """Should work with default parameters when none provided."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        # Call without params argument
        result = correlate_profiles(profile_ref, profile_comp)

        assert isinstance(result, ComparisonResults)
        assert not np.isnan(result.correlation_coefficient)


class TestCorrelateProfilesEdgeCases:
    """Edge case tests for correlate_profiles."""

    def test_constant_profile(self):
        """Constant profiles should be handled (may return NaN correlation)."""
        profile_ref = Profile(np.ones(500), pixel_size=0.5e-6)
        profile_comp = Profile(np.ones(500) * 2, pixel_size=0.5e-6)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        # Constant profiles have zero variance, correlation may be NaN
        # This is expected behavior
        assert isinstance(result, ComparisonResults)

    def test_very_short_profiles(self):
        """Very short profiles should be handled."""
        profile_ref = Profile(np.random.randn(50), pixel_size=0.5e-6)
        profile_comp = Profile(np.random.randn(50), pixel_size=0.5e-6)

        params = AlignmentParameters()

        result = correlate_profiles(profile_ref, profile_comp, params)

        assert isinstance(result, ComparisonResults)
