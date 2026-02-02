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

    # X-axis in micrometers - use each profile's own pixel size
    x_ref = np.arange(len(ref_data)) * profile_ref.pixel_size * 1e6
    x_comp = np.arange(len(comp_data)) * profile_comp.pixel_size * 1e6

    # Convert heights to micrometers
    ref_data_um = ref_data * 1e6
    comp_data_um = comp_data * 1e6

    # Calculate common y-axis limits (use nanmin/nanmax to handle NaN values)
    all_heights = np.concatenate([ref_data_um, comp_data_um])
    y_min, y_max = np.nanmin(all_heights), np.nanmax(all_heights)
    y_margin = (y_max - y_min) * 0.1
    y_limits = (y_min - y_margin, y_max + y_margin)

    # Top: Reference profile (starts at 0)
    axes[0].plot(x_ref, ref_data_um, "b-", linewidth=0.8)
    axes[0].set_xlabel("Position (μm)")
    axes[0].set_ylabel("Height (μm)")
    axes[0].set_title("Reference Profile")
    axes[0].set_xlim(0, x_ref.max())
    axes[0].set_ylim(y_limits)
    axes[0].grid(True, alpha=0.3)

    # Middle: Comparison profile (starts at 0)
    axes[1].plot(x_comp, comp_data_um, "r-", linewidth=0.8)
    axes[1].set_xlabel("Position (μm)")
    axes[1].set_ylabel("Height (μm)")
    axes[1].set_title("Comparison Profile")
    axes[1].set_xlim(0, x_comp.max())
    axes[1].set_ylim(y_limits)
    axes[1].grid(True, alpha=0.3)

    # Bottom: Aligned profiles overlaid
    # For profiles with different pixel sizes, we need to use the equalized coordinate
    # system (where both profiles have the same pixel size after resampling).
    # The position_shift is calculated in this equalized space.

    # Get the equalized pixel size (the larger of the two = lower resolution)
    equalized_pixel_size = max(profile_ref.pixel_size, profile_comp.pixel_size)

    # Create x coordinates in the equalized space
    # After equalization, physical lengths are preserved but sample counts change
    ref_physical_length_um = len(ref_data) * profile_ref.pixel_size * 1e6
    comp_physical_length_um = len(comp_data) * profile_comp.pixel_size * 1e6

    # X coordinates for equalized profiles (in micrometers)
    n_ref_equalized = int(round(ref_physical_length_um / (equalized_pixel_size * 1e6)))
    n_comp_equalized = int(
        round(comp_physical_length_um / (equalized_pixel_size * 1e6))
    )

    x_ref_eq = np.arange(n_ref_equalized) * equalized_pixel_size * 1e6
    x_comp_eq = np.arange(n_comp_equalized) * equalized_pixel_size * 1e6

    # Resample profile data to equalized coordinates if needed
    if profile_ref.pixel_size != equalized_pixel_size:
        ref_interpolator = interp1d(
            x_ref,
            ref_data_um,
            kind="linear",
            fill_value="extrapolate",  # type: ignore[arg-type]
        )
        ref_data_eq_um = ref_interpolator(x_ref_eq)
    else:
        ref_data_eq_um = ref_data_um

    if profile_comp.pixel_size != equalized_pixel_size:
        comp_interpolator = interp1d(
            x_comp,
            comp_data_um,
            kind="linear",
            fill_value="extrapolate",  # type: ignore[arg-type]
        )
        comp_data_eq_um = comp_interpolator(x_comp_eq)
    else:
        comp_data_eq_um = comp_data_um

    # Apply position_shift and scale_factor in the equalized space
    # The correlator defines position_shift as the shift of the LARGER profile.
    # Positive shift = larger profile shifts LEFT
    # Negative shift = smaller profile shifts RIGHT
    scale = result.scale_factor if not np.isnan(result.scale_factor) else 1.0
    shift_um = result.position_shift * 1e6 if not np.isnan(result.position_shift) else 0

    # Determine which profile is larger (after equalization)
    ref_is_larger = len(x_ref_eq) >= len(x_comp_eq)

    x_ref_aligned = x_ref_eq  # reference stays at origin
    if ref_is_larger:
        # Ref is larger: positive shift means ref shifts left
        # Keep ref at origin, shift comp RIGHT by shift_um
        x_comp_aligned = x_comp_eq * scale + shift_um
    else:
        # Comp is larger: positive shift means comp shifts left
        # Keep ref at origin, shift comp LEFT by shift_um
        x_comp_aligned = x_comp_eq * scale - shift_um

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

    # Determine which profile is larger (by physical length)
    comp_is_larger = comp_physical_length_um > ref_physical_length_um

    # Prepare data for plotting
    x_comp_plot = x_comp_aligned
    comp_aligned_um = comp_data_eq_um

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
    # Use the equalized profile data for the bottom panel
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
            ref_data_eq_um,
            "b-",
            linewidth=0.8,
            label="Reference",
            alpha=0.7,
        )
    else:
        # Reference is larger or equal - plot it first, then comparison on top
        axes[2].plot(
            x_ref_aligned,
            ref_data_eq_um,
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

    # Set x_limits for bottom panel based on aligned profiles
    aligned_x_max = max(x_ref_aligned.max(), x_comp_aligned.max())
    aligned_x_min = min(x_ref_aligned.min(), x_comp_aligned.min())
    aligned_x_limits = (aligned_x_min, aligned_x_max)

    axes[2].set_xlabel("Position (μm)")
    axes[2].set_ylabel("Height (μm)")
    axes[2].set_title("Aligned Profiles (after pixel equalization)")
    axes[2].set_xlim(aligned_x_limits)
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
