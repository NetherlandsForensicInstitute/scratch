"""Tests for profile correlation using real measurement data."""

from pathlib import Path

import numpy as np
import pytest

from conversion.profile_correlator import (
    AlignmentParameters,
    Profile,
    correlate_profiles,
)

PROFILES_DIR = (
    Path(__file__).parent.parent.parent
    / "resources"
    / "profile_correlator"
    / "profiles"
)

PIXEL_SIZE_M = 1.5e-6  # Sample profiles use 1.5 μm pixel size

# Expected correlation coefficients per profile pair
EXPECTED_CORRELATIONS = {
    "sample_02": 0.8542,
    "sample_03": 0.8036,
    "sample_04": 0.5762,
    "sample_05": 0.6500,
    "sample_07": 0.5636,
    "sample_08": 0.5961,
    "sample_09": 0.5608,
    "sample_11": 0.8213,
    "sample_12": 0.8379,
    "sample_14": 0.9809,
    "sample_15": 0.9802,
    "sample_16": 0.9756,
    "sample_18": 0.9365,
    "sample_19": 0.9412,
    "sample_21": 0.9624,
    "sample_22": 0.9371,
    "sample_23": 0.9813,
}


def discover_profile_pairs() -> list[tuple[str, Path, Path]]:
    """Discover all profile pairs in the profiles folder."""
    pairs = []
    for ref_path in sorted(PROFILES_DIR.glob("*_ref.npy")):
        name = ref_path.stem.replace("_ref", "")
        comp_path = ref_path.parent / f"{name}_comp.npy"
        if comp_path.exists():
            pairs.append((name, ref_path, comp_path))
    return pairs


PROFILE_PAIRS = discover_profile_pairs()


@pytest.mark.integration
class TestProfilePairs:
    """Tests for real profile pairs."""

    @pytest.mark.parametrize(
        "name,ref_path,comp_path",
        PROFILE_PAIRS,
        ids=[p[0] for p in PROFILE_PAIRS],
    )
    def test_correlation(self, name: str, ref_path: Path, comp_path: Path):
        """Test correlation matches expected value."""
        ref_data = np.load(ref_path).astype(np.float64)
        comp_data = np.load(comp_path).astype(np.float64)

        ref = Profile(heights=ref_data, pixel_size=PIXEL_SIZE_M)
        comp = Profile(heights=comp_data, pixel_size=PIXEL_SIZE_M)

        result = correlate_profiles(ref, comp, AlignmentParameters())
        assert result is not None

        expected = EXPECTED_CORRELATIONS[name]
        assert abs(result.correlation_coefficient - expected) < 0.01, (
            f"Expected {expected:.4f}, got {result.correlation_coefficient:.4f}"
        )
