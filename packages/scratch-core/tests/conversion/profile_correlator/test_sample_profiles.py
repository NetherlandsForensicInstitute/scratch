"""Tests comparing profile pairs from sample_profiles folder.

This module loads real profile data from .npy files and runs correlation
analysis with visualization output.
"""

from pathlib import Path

import numpy as np
import pytest

from conversion.profile_correlator import (
    AlignmentParameters,
    Profile,
)

from .synthetic_helpers import (
    OUTPUT_DIR,
    run_correlation_with_visualization,
)

# Path to sample profiles
SAMPLE_PROFILES_DIR = (
    Path(__file__).parent.parent.parent
    / "resources"
    / "profile_correlator"
    / "sample_profiles"
)

# Output directory for visualizations
SAMPLE_OUTPUT_DIR = OUTPUT_DIR / "sample_profiles"
SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard pixel size for sample profiles (1.5 μm)
PIXEL_SIZE_M = 1.5e-6


def discover_profile_pairs() -> list[tuple[str, Path, Path]]:
    """Discover all profile pairs in the sample_profiles folder.

    :returns: List of (name, ref_path, comp_path) tuples.
    """
    pairs = []
    ref_files = sorted(SAMPLE_PROFILES_DIR.glob("*_ref.npy"))

    for ref_path in ref_files:
        # Extract profile number (e.g., "profile02" from "profile02_ref.npy")
        name = ref_path.stem.replace("_ref", "")
        comp_path = ref_path.parent / f"{name}_comp.npy"

        if comp_path.exists():
            pairs.append((name, ref_path, comp_path))

    return pairs


# Discover all profile pairs for parametrization
PROFILE_PAIRS = discover_profile_pairs()


class TestSampleProfiles:
    """Tests for real profile pairs from sample_profiles folder."""

    @pytest.mark.parametrize(
        "name,ref_path,comp_path",
        PROFILE_PAIRS,
        ids=[p[0] for p in PROFILE_PAIRS],
    )
    def test_profile_correlation(self, name: str, ref_path: Path, comp_path: Path):
        """Test correlation of a sample profile pair.

        Loads ref and comp profiles, runs correlation, and saves visualization.

        :param name: Profile pair name (e.g., "profile02").
        :param ref_path: Path to reference profile .npy file.
        :param comp_path: Path to comparison profile .npy file.
        """
        # Load profile data
        ref_data = np.load(ref_path).astype(np.float64)
        comp_data = np.load(comp_path).astype(np.float64)

        # Create Profile objects
        profile_ref = Profile(depth_data=ref_data, pixel_size=PIXEL_SIZE_M)
        profile_comp = Profile(depth_data=comp_data, pixel_size=PIXEL_SIZE_M)

        # Run correlation with default parameters
        params = AlignmentParameters()

        result = run_correlation_with_visualization(
            profile_ref,
            profile_comp,
            params,
            title=f"Sample Profile: {name}",
            output_filename=f"{name}.png",
            output_dir=SAMPLE_OUTPUT_DIR,
        )

        # Basic sanity checks - correlation should be computed
        assert not np.isnan(result.correlation_coefficient), (
            f"Correlation coefficient is NaN for {name}"
        )

        # Overlap ratio should be positive
        assert result.overlap_ratio > 0, (
            f"Overlap ratio should be > 0, got {result.overlap_ratio} for {name}"
        )

        # Print results for inspection
        print(f"\n{name}:")
        print(f"  Correlation: {result.correlation_coefficient:.4f}")
        print(f"  Overlap ratio: {result.overlap_ratio:.4f}")
        print(f"  Position shift: {result.position_shift * 1e6:.2f} μm")
        print(f"  Scale factor: {result.scale_factor:.4f}")
        print(
            f"  Ref length: {len(ref_data)} samples ({len(ref_data) * PIXEL_SIZE_M * 1e6:.1f} μm)"
        )
        print(
            f"  Comp length: {len(comp_data)} samples ({len(comp_data) * PIXEL_SIZE_M * 1e6:.1f} μm)"
        )
