from collections.abc import Mapping
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from conversion.data_formats import Mark
from conversion.plots.data_formats import CorrelationMetrics, StriationComparisonPlots
from conversion.plots.utils import (
    figure_to_array,
    get_fig_dimensions,
    plot_profiles_on_axes,
    plot_depthmap_on_axes,
    plot_side_by_side_on_axes,
    metadata_to_table_data,
)


def plot_striation_comparison_results(
    mark_ref: Mark,
    mark_comp: Mark,
    mark_ref_aligned: Mark,
    mark_comp_aligned: Mark,
    profile_ref_aligned: Mark,
    profile_comp_aligned: Mark,
    metrics: CorrelationMetrics,
    metadata_ref: dict,
    metadata_comp: dict,
) -> StriationComparisonPlots:
    """
    Generate visualization results for striation (profile) correlation comparison.

    :param mark_ref: Reference mark after filtering/equalization.
    :param mark_comp: Compared mark after filtering/equalization.
    :param mark_ref_aligned: Reference mark after alignment (for side-by-side).
    :param mark_comp_aligned: Compared mark after alignment (for side-by-side).
    :param profile_ref_aligned: Reference profile after alignment.
    :param profile_comp_aligned: Compared profile after alignment.
    :param metrics: Correlation metrics to display in overview.
    :param metadata_ref: Metadata dict for reference profile display.
    :param metadata_comp: Metadata dict for compared profile display.
    :returns: StriationComparisonResults with all rendered images.
    """
    # Filtered mark images (with axes and colorbar)
    mark1_filtered_preview_image = plot_depthmap_with_axes(
        data=mark_ref.scan_image.data,
        scale=mark_ref.scan_image.scale_x,
        title="Filtered Reference Surface A",
    )

    mark2_filtered_preview_image = plot_depthmap_with_axes(
        data=mark_comp.scan_image.data,
        scale=mark_comp.scan_image.scale_x,
        title="Filtered Compared Surface B",
    )

    # Comparison overview
    comparison_overview = plot_comparison_overview(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_ref_aligned=mark_ref_aligned,
        mark_comp_aligned=mark_comp_aligned,
        profile_ref=profile_ref_aligned,
        profile_comp=profile_comp_aligned,
        metrics=metrics,
        metadata_ref=metadata_ref,
        metadata_comp=metadata_comp,
    )

    # Side by side
    mark1_vs_moved_mark2 = plot_side_by_side_surfaces(
        data_ref=mark_ref_aligned.scan_image.data,
        data_comp=mark_comp_aligned.scan_image.data,
        scale=mark_ref_aligned.scan_image.scale_x,
    )

    # Profile plots
    similarity_plot = plot_similarity(
        profile_ref=profile_ref_aligned.scan_image.data.flatten(),
        profile_comp=profile_comp_aligned.scan_image.data.flatten(),
        scale=profile_ref_aligned.scan_image.scale_x,
        score=metrics.score,
    )

    wavelength_correlation_plot = plot_wavelength_correlation(
        profile_ref=profile_ref_aligned.scan_image.data.flatten(),
        profile_comp=profile_comp_aligned.scan_image.data.flatten(),
        scale=profile_ref_aligned.scan_image.scale_x,
        score=metrics.score,
        quality_passbands=metrics.quality_passbands,
    )

    return StriationComparisonPlots(
        similarity_plot=similarity_plot,
        comparison_overview=comparison_overview,
        mark1_filtered_preview_image=mark1_filtered_preview_image,
        mark2_filtered_preview_image=mark2_filtered_preview_image,
        mark1_vs_moved_mark2=mark1_vs_moved_mark2,
        wavelength_plot=wavelength_correlation_plot,
    )


def plot_similarity(
    profile_ref: NDArray,
    profile_comp: NDArray,
    scale: float,
    score: float,
) -> np.ndarray:
    """
    Plot two aligned profiles overlaid (similarity plot).

    :param profile_ref: Reference profile (aligned, 1D).
    :param profile_comp: Compared profile (aligned, 1D).
    :param scale: scale of the profiles in meters.
    :param score: Pre-computed correlation coefficient from ProfileCorrelatorSingle.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    plot_profiles_on_axes(
        ax,
        profile_ref,
        profile_comp,
        scale,
        score,
        title="Similarity Score (Correlation Coefficient)",
    )
    fig.tight_layout(h_pad=3.0)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_wavelength_correlation(
    profile_ref: NDArray,
    profile_comp: NDArray,
    scale: float,
    score: float,
    quality_passbands: Mapping[tuple[float, float], float],
) -> np.ndarray:
    """
    Plot aligned profiles with wavelength-range dependent cross-correlation.

    :param profile_ref: Reference profile (aligned).
    :param profile_comp: Compared profile (aligned).
    :param scale: Scale of the profiles in meters.
    :param score: Correlation coefficient.
    :param quality_passbands: Mapping from (low, high) wavelength band in µm to correlation coefficient.
    :returns:rendered image.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Aligned profiles
    plot_profiles_on_axes(
        axes[0],
        profile_ref,
        profile_comp,
        scale,
        score,
        title="Similarity Score (Cross Coefficient), full range",
    )

    # Subplot 2: Wavelength-dependent correlation
    get_wavelength_correlation_plot(axes[1], quality_passbands)

    fig.tight_layout(h_pad=3.0)
    arr = figure_to_array(fig)
    plt.close(fig)

    return arr


def get_wavelength_correlation_plot(
    ax: Axes, quality_passbands: Mapping[tuple[float, float], float]
):
    """
    Plot correlation coefficients for different wavelength passbands.

    :param ax: Matplotlib axes to plot on.
    :param quality_passbands: Mapping from (low, high) wavelength bands in µm
        to correlation coefficients (0-1 scale).
    """
    xs = np.arange(1, len(quality_passbands) + 1)

    ax.plot(xs, [v * 100 for v in quality_passbands.values()], "b-*", linewidth=2)

    ax.set_ylim(-5, 105)
    ax.set_xlim(xs[0] - xs[-1] * 0.01, xs[-1] + xs[-1] * 0.01)
    ax.grid(True)
    ax.tick_params(labelsize=14)

    # X tick labels showing wavelength ranges
    labels = [f"{int(lo)}-{int(hi)}" for lo, hi in quality_passbands]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Wavelength ranges of geometrical details [µm]", fontsize=14)
    ax.set_ylabel("Correlation Coefficient", fontsize=14)


def plot_depthmap_with_axes(data: NDArray, scale: float, title: str) -> np.ndarray:
    """
    Plot a depth map rendering of a mark.

    :param data: data to plot in meters.
    :param scale: scale of the data in meters.
    :param title: Title for the plot.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    height, width = data.shape
    fig_height, fig_width = get_fig_dimensions(height, width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_depthmap_on_axes(ax, fig, data, scale, title, shrink_colorbar=0.5)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_side_by_side_surfaces(
    data_ref: NDArray,
    data_comp: NDArray,
    scale: float,
) -> np.ndarray:
    """
    Plot two aligned marks side by side with a small gap.

    :param data_ref: Reference data (aligned) in meters.
    :param data_comp: Compared data (aligned) in meters.
    :param scale: Scale of the data in meters.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    # Create combined data for size calculation
    gap_width = int(np.ceil(min(data_ref.shape[1], data_comp.shape[1]) / 100))
    combined_width = data_ref.shape[1] + gap_width + data_comp.shape[1]
    height = data_ref.shape[0]

    fig_height, fig_width = get_fig_dimensions(height, combined_width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_side_by_side_on_axes(ax, fig, data_ref, data_comp, scale, shrink_colorbar=0.5)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def _draw_metadata_box(
    ax: Axes, metadata: dict, title: str | None = None, draw_border: bool = True
):
    """Draw a metadata box with key-value pairs and optional title and border."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw border
    for spine in ax.spines.values():
        spine.set_visible(draw_border)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    # Title above box
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Create table data with text wrapping
    table_data = metadata_to_table_data(metadata, wrap_width=22)

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        colWidths=[0.55, 0.55],
        loc="center",
        edges="open",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Make keys bold
    for i in range(len(table_data)):
        table[i, 0].set_text_props(fontweight="bold", ha="right")
        table[i, 1].set_text_props(ha="left")


def plot_comparison_overview(
    mark_ref: Mark,
    mark_comp: Mark,
    mark_ref_aligned: Mark,
    mark_comp_aligned: Mark,
    profile_ref: Mark,
    profile_comp: Mark,
    metrics: CorrelationMetrics,
    metadata_ref: dict,
    metadata_comp: dict,
) -> np.ndarray:
    """
    Generate the main results overview figure.

    This creates a composite figure with:
    - Row 0: Metadata tables for both profiles (with boxes)
    - Row 1: Filtered surface depth maps + correlation metrics
    - Row 2: Side-by-side aligned surfaces
    - Row 3: Profile comparison plot

    :param mark_ref: Reference mark.
    :param mark_comp: Compared mark.
    :param mark_ref_aligned: Reference mark after alignment.
    :param mark_comp_aligned: Compared mark after alignment.
    :param profile_ref: Reference profile (aligned).
    :param profile_comp: Compared profile (aligned).
    :param metrics: Correlation metrics to display.
    :param metadata_ref: Metadata dict for reference profile.
    :param metadata_comp: Metadata dict for compared profile.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    fig = plt.figure(figsize=(16, 14))

    # Create grid layout: 4 rows, 3 columns
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[0.18, 0.32, 0.25, 0.25],
        width_ratios=[0.35, 0.35, 0.30],
        hspace=0.4,
        wspace=0.3,
    )

    # Row 0: Metadata tables with boxes
    ax_meta_ref = fig.add_subplot(gs[0, 0])
    _draw_metadata_box(ax_meta_ref, metadata_ref, "Reference Profile (A)")

    ax_meta_comp = fig.add_subplot(gs[0, 1])
    _draw_metadata_box(ax_meta_comp, metadata_comp, "Compared Profile (B)")

    # Row 1: Filtered surface depth maps + Results box
    ax_ref = fig.add_subplot(gs[1, 0])
    plot_depthmap_on_axes(
        ax_ref,
        fig,
        mark_ref.scan_image.data,
        mark_ref.scan_image.scale_x,
        title="Filtered Reference Surface A",
    )

    ax_comp = fig.add_subplot(gs[1, 1])
    plot_depthmap_on_axes(
        ax_comp,
        fig,
        mark_comp.scan_image.data,
        mark_comp.scan_image.scale_x,
        title="Filtered Compared Surface B",
    )

    ax_results = fig.add_subplot(gs[1, 2])
    items = {
        "Date report": datetime.now(),
        "Mark type": mark_ref.mark_type.value,
        "Correlation Coefficient": f"{metrics.score:.4f}",
        "Sq(A)": f"{metrics.sq_a:.4f} µm",
        "Sq(B)": f"{metrics.sq_b:.4f} µm",
        "Sq(B-A)": f"{metrics.sq_b_minus_a:.4f} µm",
        "Sq(B) / Sq(A)": f"{metrics.sq_ratio:.4f} %",
        "Sign. Diff. DsAB": f"{metrics.sign_diff_dsab:.2f} %",
        "Overlap": f"{metrics.overlap:.2f} %",
        "Data spacing": f"{metrics.data_spacing:.4f} µm",
        "Cutoff length low-pass filter": f"{mark_ref.meta_data.get('lowpass_cutoff'):.0f} µm",
        "Cutoff length high-pass filter": f"{mark_ref.meta_data.get('highpass_cutoff'):.0f} µm",
    }
    _draw_metadata_box(ax_results, items, draw_border=False)

    # Row 2: Side-by-side
    ax_side = fig.add_subplot(gs[2, :2])
    plot_side_by_side_on_axes(
        ax_side,
        fig,
        mark_ref_aligned.scan_image.data,
        mark_comp_aligned.scan_image.data,
        mark_ref.scan_image.scale_x,
    )

    # Row 3: Profile plot
    ax_profile = fig.add_subplot(gs[3, :])
    plot_profiles_on_axes(
        ax_profile,
        profile_ref.scan_image.data,
        profile_comp.scan_image.data,
        profile_ref.scan_image.scale_x,
        metrics.score,
        title="Reference Profile A / Moved Compared Profile B",
    )

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr
