"""Generate scatter plots of correlation vs overlap for sample profiles."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from conversion.profile_correlator import (
    AlignmentParameters,
    Profile,
    correlate_profiles,
)

# Path to sample profiles
SAMPLE_PROFILES_DIR = (
    Path(__file__).parent.parent.parent
    / "resources"
    / "profile_correlator"
    / "sample_profiles"
)

# Output directory
OUTPUT_DIR = (
    Path(__file__).parent / "outputs" / "correlate_profiles" / "sample_profiles"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard pixel size for sample profiles (1.5 μm)
PIXEL_SIZE_M = 1.5e-6

# Color coding for profile categories
PROFILE_CATEGORIES = {
    "KNM hoge CCF": {"profiles": ["02", "03", "04", "05"], "color": "red"},
    "KNM lage CCF": {"profiles": ["07", "08", "09"], "color": "orange"},
    "KM lage CCF": {"profiles": ["11", "12"], "color": "blue"},
    "KM hoge CCF": {"profiles": ["14", "15", "16"], "color": "green"},
    "Unknown": {"profiles": ["18", "19"], "color": "black"},
    "Goed uitlijning": {"profiles": ["21", "22", "23"], "color": "purple"},
}


def get_category(profile_num: str) -> tuple[str, str]:
    """Get category name and color for a profile number."""
    for category, info in PROFILE_CATEGORIES.items():
        if profile_num in info["profiles"]:
            return category, info["color"]
    return "Unknown", "gray"


def discover_profile_pairs() -> list[tuple[str, Path, Path]]:
    """Discover all profile pairs in the sample_profiles folder."""
    pairs = []
    ref_files = sorted(SAMPLE_PROFILES_DIR.glob("*_ref.npy"))

    for ref_path in ref_files:
        name = ref_path.stem.replace("_ref", "")
        comp_path = ref_path.parent / f"{name}_comp.npy"

        if comp_path.exists():
            pairs.append((name, ref_path, comp_path))

    return pairs


def main():
    """Generate scatter plots."""
    params = AlignmentParameters()
    min_overlap_um = params.min_overlap_distance * 1e6

    # Collect results
    results = []
    for name, ref_path, comp_path in discover_profile_pairs():
        ref_data = np.load(ref_path).astype(np.float64)
        comp_data = np.load(comp_path).astype(np.float64)

        profile_ref = Profile(depth_data=ref_data, pixel_size=PIXEL_SIZE_M)
        profile_comp = Profile(depth_data=comp_data, pixel_size=PIXEL_SIZE_M)

        result = correlate_profiles(profile_ref, profile_comp, params)

        profile_num = name.replace("profile", "")
        category, color = get_category(profile_num)

        results.append(
            {
                "name": name,
                "num": profile_num,
                "correlation": result.correlation_coefficient,
                "overlap_ratio": result.overlap_ratio,
                "overlap_length_um": result.overlap_length * 1e6,
                "category": category,
                "color": color,
            }
        )

    # Plot 1: Correlation vs Overlap Ratio (with numbers)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot by category for legend
    for category, info in PROFILE_CATEGORIES.items():
        cat_results = [r for r in results if r["category"] == category]
        if cat_results:
            x = [r["overlap_ratio"] for r in cat_results]
            y = [r["correlation"] for r in cat_results]
            ax.scatter(x, y, c=info["color"], s=100, label=category, alpha=0.8)

            # Add labels
            for r in cat_results:
                ax.annotate(
                    r["num"],
                    (r["overlap_ratio"], r["correlation"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9,
                )

    ax.set_xlabel("Overlap Ratio", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.set_title(
        f"Sample Profile Correlations (min_overlap = {min_overlap_um:.0f} μm)",
        fontsize=14,
    )
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_vs_overlap_summary.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'correlation_vs_overlap_summary.png'}")

    # Plot 2: Correlation vs Overlap Length in μm (with numbers)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot by category for legend
    for category, info in PROFILE_CATEGORIES.items():
        cat_results = [r for r in results if r["category"] == category]
        if cat_results:
            x = [r["overlap_length_um"] for r in cat_results]
            y = [r["correlation"] for r in cat_results]
            ax.scatter(x, y, c=info["color"], s=100, label=category, alpha=0.8)

            # Add labels
            for r in cat_results:
                ax.annotate(
                    r["num"],
                    (r["overlap_length_um"], r["correlation"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9,
                )

    # Add minimum overlap line
    ax.axvline(
        x=min_overlap_um,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Min overlap ({min_overlap_um:.0f} μm)",
    )

    ax.set_xlabel("Overlap Length (μm)", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.set_title(
        f"Sample Profile Correlations (min_overlap = {min_overlap_um:.0f} μm)",
        fontsize=14,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_vs_overlap_um_summary.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'correlation_vs_overlap_um_summary.png'}")


if __name__ == "__main__":
    main()
