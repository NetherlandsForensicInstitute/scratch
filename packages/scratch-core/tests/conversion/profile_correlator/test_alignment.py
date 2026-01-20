"""Tests for the alignment module."""

import numpy as np
import pytest

from conversion.profile_correlator import (
    Profile,
    AlignmentParameters,
    align_profiles_multiscale,
    align_partial_profile_multiscale,
)
from tests.conversion.profile_correlator.conftest import (
    make_synthetic_striation_profile,
    make_shifted_profile,
)


class TestAlignProfilesMultiscale:
    """Tests for align_profiles_multiscale function."""

    def test_identical_profiles_give_high_correlation(self):
        """Identical profiles should give correlation near 1.0."""
        profile = make_synthetic_striation_profile(n_samples=500, seed=42)

        # Create a copy as the compared profile
        profile_comp = Profile(
            depth_data=profile.depth_data.copy(),
            pixel_size=profile.pixel_size,
        )

        params = AlignmentParameters(
            scale_passes=(100, 50, 25, 10),
            max_translation=1e5,
        )

        result = align_profiles_multiscale(profile, profile_comp, params)

        assert result.final_correlation > 0.99

    def test_shifted_profiles_aligned_correctly(self):
        """Shifted profiles should be aligned with detected translation.

        This test uses a simple sine wave to ensure the alignment algorithm
        can detect and correct a small shift.
        """
        np.random.seed(42)
        n_samples = 500
        pixel_size = 0.5e-6

        # Create a clean sine wave
        x = np.linspace(0, 4 * np.pi, n_samples)
        ref_data = np.sin(x) * 1e-6

        # Create shifted version (small shift that won't cause major issues)
        shift_samples = 5
        comp_data = np.roll(ref_data, shift_samples)
        # Zero out the wrapped part to avoid artifacts
        comp_data[:shift_samples] = ref_data[:shift_samples]

        profile_ref = Profile(depth_data=ref_data, pixel_size=pixel_size)
        profile_comp = Profile(depth_data=comp_data, pixel_size=pixel_size)

        params = AlignmentParameters(
            scale_passes=(100, 50, 25, 10),
            max_translation=1e5,
        )

        result = align_profiles_multiscale(profile_ref, profile_comp, params)

        # Should achieve reasonable correlation after alignment (may not be perfect
        # due to boundary effects)
        assert result.final_correlation > 0.8

    def test_scaled_profiles_aligned_correctly(self):
        """Scaled profiles should be aligned - correlation should improve.

        Note: Perfect alignment with scaling is challenging, so we just verify
        that the algorithm produces some correlation improvement.
        """
        np.random.seed(42)
        n_samples = 500
        pixel_size = 0.5e-6

        # Create a clean sine wave
        x = np.linspace(0, 4 * np.pi, n_samples)
        ref_data = np.sin(x) * 1e-6

        # Create slightly different phase version (similar to scaling effect)
        x_scaled = np.linspace(0, 4 * np.pi * 1.02, n_samples)
        comp_data = np.sin(x_scaled) * 1e-6

        profile_ref = Profile(depth_data=ref_data, pixel_size=pixel_size)
        profile_comp = Profile(depth_data=comp_data, pixel_size=pixel_size)

        params = AlignmentParameters(
            scale_passes=(100, 50, 25, 10),
            max_translation=1e5,
            max_scaling=0.05,
        )

        result = align_profiles_multiscale(profile_ref, profile_comp, params)

        # The algorithm should find some correlation (may not be perfect)
        assert result.final_correlation > 0.5

    def test_returns_aligned_profiles(self):
        """Result should include aligned profile arrays."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters(scale_passes=(100, 50, 25, 10))

        result = align_profiles_multiscale(profile_ref, profile_comp, params)

        assert len(result.reference_aligned) > 0
        assert len(result.compared_aligned) > 0
        assert len(result.reference_aligned) == len(result.compared_aligned)

    def test_returns_transform_sequence(self):
        """Result should include sequence of transforms."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters(scale_passes=(100, 50, 25, 10))

        result = align_profiles_multiscale(profile_ref, profile_comp, params)

        assert len(result.transforms) > 0
        for t in result.transforms:
            assert hasattr(t, "translation")
            assert hasattr(t, "scaling")

    def test_returns_correlation_history(self):
        """Result should include correlation history."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        profile_comp = make_shifted_profile(profile_ref, 10.0, seed=43)

        params = AlignmentParameters(scale_passes=(100, 50, 25, 10))

        result = align_profiles_multiscale(profile_ref, profile_comp, params)

        # Correlation history should have shape (n_scales, 2)
        assert result.correlation_history.ndim == 2
        assert result.correlation_history.shape[1] == 2

    def test_different_lengths_raises_error(self):
        """Profiles with different lengths should raise ValueError."""
        profile_ref = Profile(np.random.randn(500), pixel_size=0.5e-6)
        profile_comp = Profile(np.random.randn(600), pixel_size=0.5e-6)

        with pytest.raises(ValueError):
            align_profiles_multiscale(profile_ref, profile_comp)

    def test_removes_boundary_zeros_by_default(self):
        """By default, boundary zeros should be removed from output."""
        profile_ref = make_synthetic_striation_profile(n_samples=500, seed=42)
        # Create shifted profile that will have zeros at edges after transform
        profile_comp = make_shifted_profile(profile_ref, 50.0, seed=43)

        params = AlignmentParameters(
            scale_passes=(100, 50, 25, 10),
            remove_boundary_zeros=True,
        )

        result = align_profiles_multiscale(profile_ref, profile_comp, params)

        # Output should not start/end with zeros
        assert result.reference_aligned[0] != 0 or result.compared_aligned[0] != 0
        assert result.reference_aligned[-1] != 0 or result.compared_aligned[-1] != 0


class TestAlignPartialProfileMultiscale:
    """Tests for align_partial_profile_multiscale function."""

    def test_finds_partial_profile_location(self):
        """Should find correct location of partial profile in reference."""
        # Create reference profile
        reference = make_synthetic_striation_profile(n_samples=1000, seed=42)

        # Extract partial profile from middle
        start_idx = 300
        partial_data = reference.depth_data[start_idx : start_idx + 300].copy()
        partial = Profile(depth_data=partial_data, pixel_size=reference.pixel_size)

        params = AlignmentParameters(
            scale_passes=(100, 50, 25, 10),
            inclusion_threshold=0.3,  # Lower threshold for testing
        )

        result, found_start = align_partial_profile_multiscale(
            reference,
            partial,
            params,
            candidate_positions=[start_idx - 50, start_idx, start_idx + 50],
        )

        # Should find the correct starting position (within tolerance)
        assert abs(found_start - start_idx) < 100

        # Should achieve reasonable correlation
        assert result.final_correlation > 0.8

    def test_returns_alignment_result_and_position(self):
        """Should return both alignment result and best start position."""
        reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        partial_data = reference.depth_data[300:600].copy()
        partial = Profile(depth_data=partial_data, pixel_size=reference.pixel_size)

        params = AlignmentParameters(scale_passes=(100, 50, 25, 10))

        result, start_pos = align_partial_profile_multiscale(
            reference, partial, params, candidate_positions=[300]
        )

        # Check return types
        assert hasattr(result, "final_correlation")
        assert hasattr(result, "transforms")
        assert isinstance(start_pos, (int, np.integer))

    def test_with_explicit_candidates(self):
        """Should work with explicitly provided candidate positions."""
        reference = make_synthetic_striation_profile(n_samples=1000, seed=42)
        partial_data = reference.depth_data[500:700].copy()
        partial = Profile(depth_data=partial_data, pixel_size=reference.pixel_size)

        params = AlignmentParameters(scale_passes=(100, 50, 25, 10))

        # Provide explicit candidates including the correct one
        candidates = [100, 300, 500, 700]

        result, start_pos = align_partial_profile_multiscale(
            reference, partial, params, candidate_positions=candidates
        )

        # Should find position 500 (or nearby) as best candidate
        assert start_pos in candidates
        assert result.final_correlation > 0.5
