"""Helper functions for synthetic profile correlation tests.

This module provides functions for creating synthetic striation profiles
and visualizing correlation results.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from conversion.profile_correlator import (
    AlignmentParameters,
    ComparisonResults,
    Profile,
    correlate_profiles,
)

# Output directory for test visualizations
OUTPUT_DIR = Path(__file__).parent / "outputs" / "correlate_profiles"


def create_base_profile(
    n_samples: int = 1000,
    amplitude_m: float = 0.5e-6,
    noise_level: float = 0.02,
    seed: int = 42,
) -> NDArray[np.floating]:
    """Generate a realistic striation profile using multiple sine frequencies.

    Creates a unique pattern that mimics striation marks with ridges and valleys.

    :param n_samples: Number of samples in the profile.
    :param amplitude_m: Base amplitude in meters.
    :param noise_level: Relative noise level (fraction of amplitude).
    :param seed: Random seed for reproducibility.
    :returns: 1D array of height values in meters.
    """
    np.random.seed(seed)

    x = np.linspace(0, 20 * np.pi, n_samples)

    # Primary striation pattern with multiple frequencies
    data = np.sin(x) * amplitude_m
    data += np.sin(2.3 * x) * amplitude_m * 0.4
    data += np.sin(0.7 * x) * amplitude_m * 0.3
    data += np.sin(5.1 * x) * amplitude_m * 0.15

    # Add Gaussian noise
    noise = np.random.normal(0, amplitude_m * noise_level, n_samples)
    data += noise

    return data


def create_shifted_profiles(
    base: NDArray[np.floating],
    shift_samples: int,
    pixel_size_m: float = 1.5e-6,
    noise_seed: int = 123,
) -> tuple[Profile, Profile]:
    """Create two profiles where comparison is shifted relative to reference.

    Creates an extended profile and extracts two windows at different positions,
    ensuring genuine overlap without wrap-around artifacts.

    :param base: Base profile array (used to determine pattern characteristics).
    :param shift_samples: Number of samples to shift the comparison profile.
    :param pixel_size_m: Pixel size in meters.
    :param noise_seed: Random seed for small noise addition.
    :returns: Tuple of (reference_profile, comparison_profile).
    """
    np.random.seed(noise_seed)
    n = len(base)

    # Create an extended profile long enough to extract two shifted windows
    extended_length = n + shift_samples
    amplitude_m = np.std(base) * 2

    # Generate extended pattern using same frequency components as base profile
    x_extended = np.linspace(0, 20 * np.pi * extended_length / n, extended_length)
    extended = np.sin(x_extended) * amplitude_m
    extended += np.sin(2.3 * x_extended) * amplitude_m * 0.4
    extended += np.sin(0.7 * x_extended) * amplitude_m * 0.3
    extended += np.sin(5.1 * x_extended) * amplitude_m * 0.15

    # Extract reference window from start
    ref_data = extended[:n].copy()

    # Extract comparison window shifted by shift_samples
    comp_data = extended[shift_samples : shift_samples + n].copy()

    # Add small independent noise to each
    noise_level = amplitude_m * 0.02
    ref_data += np.random.normal(0, noise_level, n)
    comp_data += np.random.normal(0, noise_level, n)

    profile_ref = Profile(depth_data=ref_data, pixel_size=pixel_size_m)
    profile_comp = Profile(depth_data=comp_data, pixel_size=pixel_size_m)

    return profile_ref, profile_comp


def create_partial_length_profiles(
    base: NDArray[np.floating],
    partial_ratio: float,
    pixel_size_m: float = 1.5e-6,
) -> tuple[Profile, Profile]:
    """Create profiles where comparison is a subset of reference.

    Reference is the full profile, comparison is an extracted subset.

    :param base: Base profile array.
    :param partial_ratio: Length of comparison as fraction of reference (e.g., 0.5 for 50%).
    :param pixel_size_m: Pixel size in meters.
    :returns: Tuple of (reference_profile, comparison_profile).
    """
    n = len(base)
    partial_length = int(n * partial_ratio)

    # Extract from middle of the profile
    start_idx = (n - partial_length) // 2
    end_idx = start_idx + partial_length

    ref_data = base.copy()
    comp_data = base[start_idx:end_idx].copy()

    profile_ref = Profile(depth_data=ref_data, pixel_size=pixel_size_m)
    profile_comp = Profile(depth_data=comp_data, pixel_size=pixel_size_m)

    return profile_ref, profile_comp


def create_scaled_profiles(
    base: NDArray[np.floating],
    scale_factor: float,
    pixel_size_m: float = 1.5e-6,
) -> tuple[Profile, Profile]:
    """Create profiles where comparison is stretched relative to reference.

    Uses scipy.interpolate.interp1d for smooth stretching.

    :param base: Base profile array.
    :param scale_factor: Scaling factor (e.g., 1.05 for 5% stretch).
    :param pixel_size_m: Pixel size in meters.
    :returns: Tuple of (reference_profile, comparison_profile).
    """
    n = len(base)

    # Reference is the original
    ref_data = base.copy()

    # Create scaled version using interpolation
    x_orig = np.arange(n)
    interpolator = interp1d(x_orig, base, kind="cubic", fill_value="extrapolate")  # type: ignore[arg-type]

    # New x coordinates: stretch by scale_factor
    # If scale_factor > 1, the pattern is stretched (more samples per period)
    x_scaled = np.arange(n) / scale_factor
    comp_data = interpolator(x_scaled)

    profile_ref = Profile(depth_data=ref_data, pixel_size=pixel_size_m)
    profile_comp = Profile(depth_data=comp_data, pixel_size=pixel_size_m)

    return profile_ref, profile_comp


def plot_correlation_result(
    profile_ref: Profile,
    profile_comp: Profile,
    result: ComparisonResults,
    title: str,
    output_path: Path,
) -> None:
    """Create visualization with 3 subplots showing correlation result.

    - Top: Input reference profile
    - Middle: Input comparison profile (at original position for partial profiles)
    - Bottom: Aligned profiles overlaid with metrics annotation and green overlap region

    :param profile_ref: Reference profile.
    :param profile_comp: Comparison profile.
    :param result: Comparison results from correlate_profiles.
    :param title: Title for the figure.
    :param output_path: Path to save the PNG file.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Get data
    ref_data = profile_ref.mean_profile()
    comp_data = profile_comp.mean_profile()
    pixel_size = profile_ref.pixel_size

    # X-axis in micrometers
    x_ref = np.arange(len(ref_data)) * pixel_size * 1e6
    x_comp = np.arange(len(comp_data)) * pixel_size * 1e6

    # Convert heights to micrometers
    ref_data_um = ref_data * 1e6
    comp_data_um = comp_data * 1e6

    # For partial length profiles, calculate original position of comparison within reference
    # The comparison was extracted from the middle of the longer profile
    if len(ref_data) != len(comp_data):
        if len(ref_data) < len(comp_data):
            # ref is short, comp is long - ref was extracted from middle of comp
            n_long = len(comp_data)
            n_short = len(ref_data)
            original_start_idx = (n_long - n_short) // 2
            # For display: show ref at its original position in comp's coordinate system
            comp_original_start_um = 0  # comp starts at 0
            ref_original_start_um = original_start_idx * pixel_size * 1e6
        else:
            # ref is long, comp is short - comp was extracted from middle of ref
            n_long = len(ref_data)
            n_short = len(comp_data)
            original_start_idx = (n_long - n_short) // 2
            comp_original_start_um = original_start_idx * pixel_size * 1e6
            ref_original_start_um = 0  # ref starts at 0
    else:
        comp_original_start_um = 0
        ref_original_start_um = 0

    # Calculate common axis limits
    all_heights = np.concatenate([ref_data_um, comp_data_um])
    y_min, y_max = all_heights.min(), all_heights.max()
    y_margin = (y_max - y_min) * 0.1
    y_limits = (y_min - y_margin, y_max + y_margin)

    # X-axis limits: use the longer profile's extent
    # Account for both original positions
    x_max = max(
        x_ref.max() + ref_original_start_um, x_comp.max() + comp_original_start_um
    )
    x_limits = (0, x_max)

    # Top: Reference profile
    axes[0].plot(x_ref + ref_original_start_um, ref_data_um, "b-", linewidth=0.8)
    axes[0].set_xlabel("Position (μm)")
    axes[0].set_ylabel("Height (μm)")
    axes[0].set_title("Reference Profile")
    axes[0].set_xlim(x_limits)
    axes[0].set_ylim(y_limits)
    axes[0].grid(True, alpha=0.3)

    # Middle: Comparison profile (at original position for partial profiles)
    axes[1].plot(x_comp + comp_original_start_um, comp_data_um, "r-", linewidth=0.8)
    axes[1].set_xlabel("Position (μm)")
    axes[1].set_ylabel("Height (μm)")
    axes[1].set_title("Comparison Profile")
    axes[1].set_xlim(x_limits)
    axes[1].set_ylim(y_limits)
    axes[1].grid(True, alpha=0.3)

    # Bottom: Aligned profiles overlaid
    # Determine alignment position using position_shift
    scale = result.scale_factor if not np.isnan(result.scale_factor) else 1.0

    # Use position_shift for alignment
    shift_um = result.position_shift * 1e6 if not np.isnan(result.position_shift) else 0
    x_comp_aligned = x_comp * scale + shift_um
    x_ref_aligned = x_ref  # reference stays at origin

    # Calculate overlap region for green highlighting
    overlap_length_um = (
        result.overlap_length * 1e6 if not np.isnan(result.overlap_length) else 0
    )

    # Calculate overlap from position_shift
    # Overlap region is where both profiles have data
    ref_start, ref_end = x_ref_aligned.min(), x_ref_aligned.max()
    comp_start, comp_end = x_comp_aligned.min(), x_comp_aligned.max()
    overlap_start_um = max(ref_start, comp_start)
    overlap_end_um = min(ref_end, comp_end)

    # Determine which profile is larger (by number of samples)
    comp_is_larger = len(comp_data) > len(ref_data)

    # Prepare aligned comparison data for plotting
    if abs(scale - 1.0) > 0.001:
        # Interpolate the comparison data to show aligned heights
        comp_interpolator = interp1d(
            x_comp,
            comp_data_um,
            kind="cubic",
            fill_value="extrapolate",  # type: ignore[arg-type]
        )
        # Sample at positions that correspond to the aligned reference x after inverse transform
        x_sample = (x_ref_aligned - shift_um) / scale
        # Only use samples within the original comparison range
        valid_mask = (x_sample >= x_comp.min()) & (x_sample <= x_comp.max())
        x_comp_plot = x_ref_aligned[valid_mask]
        comp_aligned_um = comp_interpolator(x_sample[valid_mask])
    else:
        x_comp_plot = x_comp_aligned
        comp_aligned_um = comp_data_um

    # Add green background for overlap region (draw before the lines)
    if overlap_length_um > 0:
        axes[2].axvspan(
            overlap_start_um,
            overlap_end_um,
            alpha=0.3,
            color="green",
            label="Overlap region",
        )

    # Plot profiles: larger one first (background), smaller one on top
    if comp_is_larger:
        # Comparison is larger - plot it first, then reference on top
        axes[2].plot(
            x_comp_plot,
            comp_aligned_um,
            "r-",
            linewidth=0.8,
            label="Comparison (aligned)",
            alpha=0.7,
        )
        axes[2].plot(
            x_ref_aligned,
            ref_data_um,
            "b-",
            linewidth=0.8,
            label="Reference",
            alpha=0.7,
        )
    else:
        # Reference is larger or equal - plot it first, then comparison on top
        axes[2].plot(
            x_ref_aligned,
            ref_data_um,
            "b-",
            linewidth=0.8,
            label="Reference",
            alpha=0.7,
        )
        axes[2].plot(
            x_comp_plot,
            comp_aligned_um,
            "r-",
            linewidth=0.8,
            label="Comparison (aligned)",
            alpha=0.7,
        )

    axes[2].set_xlabel("Position (μm)")
    axes[2].set_ylabel("Height (μm)")
    axes[2].set_title("Aligned Profiles")
    axes[2].set_xlim(x_limits)
    axes[2].set_ylim(y_limits)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = (
        f"Correlation: {result.correlation_coefficient:.4f}\n"
        f"Overlap ratio: {result.overlap_ratio:.4f}\n"
        f"Scale factor: {result.scale_factor:.4f}\n"
        f"Position shift: {result.position_shift * 1e6:.2f} μm"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    axes[2].text(
        0.02,
        0.98,
        metrics_text,
        transform=axes[2].transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_correlation_with_visualization(
    profile_ref: Profile,
    profile_comp: Profile,
    params: AlignmentParameters,
    title: str,
    output_filename: str,
) -> ComparisonResults:
    """Run correlation and generate visualization.

    Wrapper function that calls correlate_profiles and creates a visualization.

    :param profile_ref: Reference profile.
    :param profile_comp: Comparison profile.
    :param params: Alignment parameters.
    :param title: Title for the visualization.
    :param output_filename: Filename for the output PNG (without directory).
    :returns: ComparisonResults from correlate_profiles.
    """
    result = correlate_profiles(profile_ref, profile_comp, params)

    output_path = OUTPUT_DIR / output_filename
    plot_correlation_result(profile_ref, profile_comp, result, title, output_path)

    return result
