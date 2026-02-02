"""Generate visualization images for all MATLAB test cases.

This script loads each MATLAB test case from resources/profile_correlator/,
runs the correlator, and generates visualization images.
"""

import json
from pathlib import Path

import numpy as np

from conversion.profile_correlator import (
    AlignmentParameters,
    Profile,
    correlate_profiles,
)

from synthetic_helpers import plot_correlation_result

# Directories
RESOURCES_DIR = Path(__file__).parent.parent.parent / "resources" / "profile_correlator"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "matlab_test_cases"


def load_matlab_test_case(case_dir: Path) -> tuple[Profile, Profile, dict]:
    """Load a MATLAB test case from disk.

    :param case_dir: Directory containing the test case files.
    :returns: Tuple of (profile_ref, profile_comp, metadata).
    """
    # Load profile data
    ref_data = np.load(case_dir / "input_profile_ref.npy")
    comp_data = np.load(case_dir / "input_profile_comp.npy")

    # Load metadata
    with open(case_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Get pixel sizes (metadata stores in micrometers, convert to meters)
    ref_pixel_size = metadata["ref_xdim"]  # Already in meters based on the values
    comp_pixel_size = metadata["comp_xdim"]

    # Create Profile objects
    profile_ref = Profile(depth_data=ref_data, pixel_size=ref_pixel_size)
    profile_comp = Profile(depth_data=comp_data, pixel_size=comp_pixel_size)

    return profile_ref, profile_comp, metadata


def generate_visualization(case_name: str, case_dir: Path, output_dir: Path) -> None:
    """Generate visualization for a single test case.

    :param case_name: Name of the test case.
    :param case_dir: Directory containing the test case files.
    :param output_dir: Directory to save the output image.
    """
    print(f"Processing: {case_name}")

    # Load test case
    profile_ref, profile_comp, metadata = load_matlab_test_case(case_dir)

    # Run correlation
    params = AlignmentParameters(
        scale_passes=(1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6),
    )

    result = correlate_profiles(profile_ref, profile_comp, params)

    # Generate visualization
    output_path = output_dir / f"{case_name}.png"
    title = f"MATLAB Test: {case_name.replace('_', ' ').title()}"

    plot_correlation_result(profile_ref, profile_comp, result, title, output_path)

    print(f"  -> Saved: {output_path.name}")
    print(f"     Correlation: {result.correlation_coefficient:.4f}")
    print(f"     Overlap ratio: {result.overlap_ratio:.4f}")


def main():
    """Generate visualizations for all MATLAB test cases."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all test case directories
    test_cases = sorted(
        [
            d
            for d in RESOURCES_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("__")
        ]
    )

    print(f"Found {len(test_cases)} test cases")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for case_dir in test_cases:
        try:
            generate_visualization(case_dir.name, case_dir, OUTPUT_DIR)
        except Exception as e:
            print(f"  ERROR: {e}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
