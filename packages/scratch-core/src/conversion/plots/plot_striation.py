from collections.abc import Mapping
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from container_models.base import FloatArray2D, ImageRGB
from conversion.data_formats import Mark
from conversion.plots.data_formats import CorrelationMetrics, StriationComparisonPlots
from conversion.plots.utils import (
    figure_to_array,
    get_figure_dimensions,
    plot_profiles_on_axes,
    plot_depth_map_on_axes,
    plot_side_by_side_on_axes,
    metadata_to_table_data,
    get_col_widths,
    get_height_ratios,
    get_metadata_dimensions,
    get_bounding_box,
)


def plot_striation_comparison_results(
    mark_reference: Mark,
    mark_compared: Mark,
    mark_reference_aligned: Mark,
    mark_compared_aligned: Mark,
    mark_profile_reference_aligned: Mark,
    mark_profile_compared_aligned: Mark,
    metrics: CorrelationMetrics,
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
) -> StriationComparisonPlots:
    """
    Generate visualization results for striation (profile) correlation comparison.

    :param mark_reference: Reference mark after filtering/equalization.
    :param mark_compared: Compared mark after filtering/equalization.
    :param mark_reference_aligned: Reference mark after alignment (for side-by-side).
    :param mark_compared_aligned: Compared mark after alignment (for side-by-side).
    :param mark_profile_reference_aligned: Reference profile after alignment.
    :param mark_profile_compared_aligned: Compared profile after alignment.
    :param metrics: Correlation metrics to display in overview.
    :param metadata_reference: Metadata dict for reference profile display.
    :param metadata_compared: Metadata dict for compared profile display.
    :returns: StriationComparisonPlots with all rendered images as arrays.
    """
    # Filtered mark images (with axes and colorbar)
    mark1_filtered_preview_image = plot_depth_map_with_axes(
        data=mark_reference.scan_image.data,
        scale=mark_reference.scan_image.scale_x,
        title="Filtered Reference Surface A",
    )

    mark2_filtered_preview_image = plot_depth_map_with_axes(
        data=mark_compared.scan_image.data,
        scale=mark_compared.scan_image.scale_x,
        title="Filtered Compared Surface B",
    )

    # Comparison overview
    comparison_overview = plot_comparison_overview(
        mark_reference=mark_reference,
        mark_compared=mark_compared,
        mark_reference_aligned=mark_reference_aligned,
        mark_compared_aligned=mark_compared_aligned,
        mark_profile_reference=mark_profile_reference_aligned,
        mark_profile_compared=mark_profile_compared_aligned,
        metrics=metrics,
        metadata_reference=metadata_reference,
        metadata_compared=metadata_compared,
    )

    # Side by side
    mark1_vs_moved_mark2 = plot_side_by_side_surfaces(
        data_reference=mark_reference_aligned.scan_image.data,
        data_compared=mark_compared_aligned.scan_image.data,
        scale=mark_reference_aligned.scan_image.scale_x,
    )

    # Profile plots
    similarity_plot = plot_similarity(
        profile_reference=mark_profile_reference_aligned.scan_image.data.flatten(),
        profile_compared=mark_profile_compared_aligned.scan_image.data.flatten(),
        scale=mark_profile_reference_aligned.scan_image.scale_x,
        score=metrics.score,
    )

    wavelength_correlation_plot = plot_wavelength_correlation(
        profile_reference=mark_profile_reference_aligned.scan_image.data.flatten(),
        profile_compared=mark_profile_compared_aligned.scan_image.data.flatten(),
        scale=mark_profile_reference_aligned.scan_image.scale_x,
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
    profile_reference: FloatArray2D,
    profile_compared: FloatArray2D,
    scale: float,
    score: float,
) -> ImageRGB:
    """
    Plot two aligned profiles overlaid (similarity plot).

    :param profile_reference: Reference profile (aligned, 1D).
    :param profile_compared: Compared profile (aligned, 1D).
    :param scale: scale of the profiles in meters.
    :param score: Pre-computed correlation coefficient from ProfileCorrelatorSingle.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    plot_profiles_on_axes(
        ax,
        profile_reference,
        profile_compared,
        scale,
        score,
        title="Similarity Score (Correlation Coefficient)",
    )
    fig.tight_layout(h_pad=3.0)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_wavelength_correlation(
    profile_reference: FloatArray2D,
    profile_compared: FloatArray2D,
    scale: float,
    score: float,
    quality_passbands: Mapping[tuple[float, float], float],
) -> ImageRGB:
    """
    Plot aligned profiles with wavelength-range dependent cross-correlation.

    :param profile_reference: Reference profile (aligned).
    :param profile_compared: Compared profile (aligned).
    :param scale: Scale of the profiles in meters.
    :param score: Correlation coefficient.
    :param quality_passbands: Mapping from (low, high) wavelength band in µm to correlation coefficient.
    :returns:rendered image.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Aligned profiles
    plot_profiles_on_axes(
        axes[0],
        profile_reference,
        profile_compared,
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

    ax.plot(xs, list(quality_passbands.values()), "b-*", linewidth=2)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(xs[0] - xs[-1] * 0.01, xs[-1] + xs[-1] * 0.01)
    ax.grid(True)
    ax.tick_params(labelsize=14)

    # X tick labels showing wavelength ranges
    labels = [f"{int(lo)}-{int(hi)}" for lo, hi in quality_passbands]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Wavelength ranges of geometrical details [µm]", fontsize=14)
    ax.set_ylabel("Correlation Coefficient", fontsize=14)


def plot_depth_map_with_axes(data: FloatArray2D, scale: float, title: str) -> ImageRGB:
    """
    Plot a depth map rendering of a mark.

    :param data: data to plot in meters.
    :param scale: scale of the data in meters.
    :param title: Title for the plot.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    height, width = data.shape
    fig_height, fig_width = get_figure_dimensions(height, width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_depth_map_on_axes(ax, fig, data, scale, title)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_side_by_side_surfaces(
    data_reference: FloatArray2D,
    data_compared: FloatArray2D,
    scale: float,
) -> ImageRGB:
    """
    Plot two aligned marks side by side with a small gap.

    :param data_reference: Reference data (aligned) in meters.
    :param data_compared: Compared data (aligned) in meters.
    :param scale: Scale of the data in meters.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    # Create combined data for size calculation
    gap_width = int(np.ceil(min(data_reference.shape[1], data_compared.shape[1]) / 100))
    combined_width = data_reference.shape[1] + gap_width + data_compared.shape[1]
    height = data_reference.shape[0]

    fig_height, fig_width = get_figure_dimensions(height, combined_width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_side_by_side_on_axes(ax, fig, data_reference, data_compared, scale)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def _draw_metadata_box(
    ax: Axes,
    metadata: dict,
    title: str | None = None,
    draw_border: bool = True,
    wrap_width: int = 25,
    side_margin: float = 0.06,
):
    """Draw a metadata box with key-value pairs."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(draw_border)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    table_data = metadata_to_table_data(metadata, wrap_width=wrap_width)
    col_widths = get_col_widths(side_margin, table_data)
    bounding_box = get_bounding_box(side_margin, table_data)

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        colWidths=col_widths,
        loc="upper center",
        edges="open",
        bbox=bounding_box,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i in range(len(table_data)):
        table[i, 0].set_text_props(fontweight="bold", ha="right")
        table[i, 0].PAD = 0.02
        table[i, 1].set_text_props(ha="left")
        table[i, 1].PAD = 0.02


def plot_comparison_overview(
    mark_reference: Mark,
    mark_compared: Mark,
    mark_reference_aligned: Mark,
    mark_compared_aligned: Mark,
    mark_profile_reference: Mark,
    mark_profile_compared: Mark,
    metrics: CorrelationMetrics,
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
    wrap_width: int = 25,
) -> ImageRGB:
    """Generate the main results overview figure with dynamic sizing."""

    # Build results metadata
    results_items = {
        "Date report": datetime.now().strftime("%Y-%m-%d"),
        "Mark type": mark_reference.mark_type.value,
        "Correlation Coefficient": f"{metrics.score:.4f}",
        "Sq(A)": f"{metrics.sq_a:.4f} µm",
        "Sq(B)": f"{metrics.sq_b:.4f} µm",
        "Sq(B-A)": f"{metrics.sq_b_minus_a:.4f} µm",
        "Sq(B) / Sq(A)": f"{metrics.sq_ratio:.4f} %",
        "Sign. Diff. DsAB": f"{metrics.sign_diff_dsab:.2f} %",
        "Overlap": f"{metrics.overlap:.2f} %",
        "Data spacing": f"{metrics.data_spacing:.4f} µm",
        "Cutoff length low-pass filter": f"{val:.0f} µm"
        if (val := mark_reference.meta_data.get("lowpass_cutoff")) is not None
        else "N/A",
        "Cutoff length high-pass filter": f"{val:.0f} µm"
        if (val := mark_reference.meta_data.get("highpass_cutoff")) is not None
        else "N/A",
    }

    max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
        metadata_compared, metadata_reference, wrap_width
    )
    height_ratios = get_height_ratios(metadata_height_ratio)

    # Adjust figure height based on content
    fig_height = 13 + (max_metadata_rows * 0.12)
    fig_height = max(12, min(16, fig_height))

    fig = plt.figure(figsize=(14, fig_height))

    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=height_ratios,
        width_ratios=[0.35, 0.35, 0.30],
        hspace=0.35,
        wspace=0.25,
    )

    # Row 0: Metadata tables
    ax_meta_reference = fig.add_subplot(gs[0, 0])
    _draw_metadata_box(
        ax_meta_reference,
        metadata_reference,
        "Reference Profile (A)",
        wrap_width=wrap_width,
    )

    ax_meta_compared = fig.add_subplot(gs[0, 1])
    _draw_metadata_box(
        ax_meta_compared,
        metadata_compared,
        "Compared Profile (B)",
        wrap_width=wrap_width,
    )

    # Row 1: Filtered surfaces + Results
    ax_reference = fig.add_subplot(gs[1, 0])
    plot_depth_map_on_axes(
        ax_reference,
        fig,
        mark_reference.scan_image.data,
        mark_reference.scan_image.scale_x,
        title="Filtered Reference Surface A",
    )

    ax_compared = fig.add_subplot(gs[1, 1])
    plot_depth_map_on_axes(
        ax_compared,
        fig,
        mark_compared.scan_image.data,
        mark_compared.scan_image.scale_x,
        title="Filtered Compared Surface B",
    )

    ax_results = fig.add_subplot(gs[1, 2])
    _draw_metadata_box(
        ax_results, results_items, draw_border=False, wrap_width=wrap_width
    )

    # Row 2: Side-by-side
    ax_side = fig.add_subplot(gs[2, :2])
    plot_side_by_side_on_axes(
        ax_side,
        fig,
        mark_reference_aligned.scan_image.data,
        mark_compared_aligned.scan_image.data,
        mark_reference.scan_image.scale_x,
    )

    # Row 3: Profile plot
    ax_profile = fig.add_subplot(gs[3, :])
    plot_profiles_on_axes(
        ax_profile,
        mark_profile_reference.scan_image.data,
        mark_profile_compared.scan_image.data,
        mark_profile_reference.scan_image.scale_x,
        metrics.score,
        title="Reference Profile A / Moved Compared Profile B. Correlation Coefficient",
    )

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.8)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.06)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr
