from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
from scipy.constants import mega

from container_models.base import FloatArray2D, ImageRGB, StriationProfile
from conversion.data_formats import Mark, MarkMetadata
from conversion.plots.comparison.data_formats import StriationComparisonPlots
from conversion.profile_correlator import StriationComparisonResults, Profile
from conversion.plots.utils import (
    finish_overview,
    render_single_panel,
    get_figure_dimensions,
    get_height_ratios,
    overview_figure_height,
    side_by_side_gap_width,
)
from conversion.plots.on_axes import (
    plot_profiles_on_axes,
    plot_side_by_side_on_axes,
    plot_depth_map_on_axes,
    plot_depth_map_with_axes,
)
from conversion.plots.metadata_tables import (
    get_metadata_dimensions,
    draw_metadata_box,
    draw_metadata_pair,
)


def plot_striation_comparison_results(
    mark_reference: Mark,
    mark_compared: Mark,
    mark_reference_aligned: Mark,
    mark_compared_aligned: Mark,
    profile_reference_aligned: Profile,
    profile_compared_aligned: Profile,
    metrics: StriationComparisonResults,
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
) -> StriationComparisonPlots:
    """
    Generate visualization results for striation (profile) correlation comparison.

    :param mark_reference: Reference mark after filtering/equalization.
    :param mark_compared: Compared mark after filtering/equalization.
    :param mark_reference_aligned: Reference mark after alignment (for side-by-side).
    :param mark_compared_aligned: Compared mark after alignment (for side-by-side).
    :param profile_reference_aligned: Reference profile after alignment.
    :param profile_compared_aligned: Compared profile after alignment.
    :param metrics: Correlation metrics to display in overview.
    :param metadata_reference: Metadata dict for reference profile display.
    :param metadata_compared: Metadata dict for compared profile display.
    :returns: StriationComparisonPlots with all rendered images as arrays.
    """
    filtered_reference_heatmap = plot_depth_map_with_axes(
        data=mark_reference.scan_image.data,
        scale=mark_reference.scan_image.scale_x,
        title="Filtered Reference Surface A",
    )

    filtered_compared_heatmap = plot_depth_map_with_axes(
        data=mark_compared.scan_image.data,
        scale=mark_compared.scan_image.scale_x,
        title="Filtered Compared Surface B",
    )

    # Comparison overview
    comparison_overview = plot_striation_comparison_overview(
        mark_reference=mark_reference,
        mark_compared=mark_compared,
        mark_reference_aligned=mark_reference_aligned,
        mark_compared_aligned=mark_compared_aligned,
        profile_reference=profile_reference_aligned,
        profile_compared=profile_compared_aligned,
        metrics=metrics,
        metadata_reference=metadata_reference,
        metadata_compared=metadata_compared,
    )

    side_by_side_heatmap = plot_side_by_side_surfaces(
        data_reference=mark_reference_aligned.scan_image.data,
        data_compared=mark_compared_aligned.scan_image.data,
        scale=mark_reference_aligned.scan_image.scale_x,
    )

    # Profile plots
    similarity_plot = plot_similarity(
        profile_reference=profile_reference_aligned.heights,
        profile_compared=profile_compared_aligned.heights,
        scale=profile_reference_aligned.pixel_size,
        score=metrics.correlation_coefficient,
    )

    return StriationComparisonPlots(
        similarity_plot=similarity_plot,
        comparison_overview=comparison_overview,
        filtered_reference_heatmap=filtered_reference_heatmap,
        filtered_compared_heatmap=filtered_compared_heatmap,
        side_by_side_heatmap=side_by_side_heatmap,
    )


def plot_similarity(
    profile_reference: StriationProfile,
    profile_compared: StriationProfile,
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
    return render_single_panel(
        (10, 4),
        lambda fig, ax: plot_profiles_on_axes(
            ax,
            profile_reference,
            profile_compared,
            scale,
            score,
            title="Similarity Score (Correlation Coefficient)",
        ),
        tight_layout_kwargs={"h_pad": 3.0},
    )


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
    gap_width = side_by_side_gap_width(data_reference, data_compared)
    combined_width = data_reference.shape[1] + gap_width + data_compared.shape[1]
    height = data_reference.shape[0]

    fig_height, fig_width = get_figure_dimensions(height, combined_width)

    return render_single_panel(
        (fig_width, fig_height),
        lambda fig, ax: plot_side_by_side_on_axes(
            ax, fig, data_reference, data_compared, scale
        ),
    )


def plot_striation_comparison_overview(
    mark_reference: Mark,
    mark_compared: Mark,
    mark_reference_aligned: Mark,
    mark_compared_aligned: Mark,
    profile_reference: Profile,
    profile_compared: Profile,
    metrics: StriationComparisonResults,
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
    wrap_width: int = 25,
) -> ImageRGB:
    """Generate the main results overview figure with dynamic sizing."""

    # Build results metadata
    results_items = build_striation_results_metadata(mark_reference, metrics)

    max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
        metadata_compared, metadata_reference, wrap_width
    )
    height_ratios = get_height_ratios(metadata_height_ratio, 0.32, 0.22, 0.20)

    fig_height = overview_figure_height(max_metadata_rows, 13, 12, 16)

    fig = plt.figure(figsize=(14, fig_height))

    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=height_ratios,
        width_ratios=[0.35, 0.35, 0.30],
        hspace=0.35,
        wspace=0.25,
    )
    # Row 0 spans full width as two equal metadata columns.
    gs_meta = gs[0, :].subgridspec(1, 2, wspace=0.15)

    # Layout — row 0: metadata pair; row 1: two filtered surfaces + results;
    # row 2: side-by-side; row 3: profile plot.
    ax_meta_reference = fig.add_subplot(gs_meta[0, 0])
    ax_meta_compared = fig.add_subplot(gs_meta[0, 1])
    ax_reference = fig.add_subplot(gs[1, 0])
    ax_compared = fig.add_subplot(gs[1, 1])
    ax_results = fig.add_subplot(gs[1, 2])
    ax_side = fig.add_subplot(gs[2, :2])
    ax_profile = fig.add_subplot(gs[3, :])

    # Row 0: metadata tables
    draw_metadata_pair(
        ax_meta_reference,
        ax_meta_compared,
        metadata_reference,
        metadata_compared,
        noun="Profile",
        wrap_width=wrap_width,
    )

    # Row 1: filtered surfaces + results metadata
    plot_depth_map_on_axes(
        ax_reference,
        fig,
        mark_reference.scan_image.data,
        mark_reference.scan_image.scale_x,
        title="Filtered Reference Surface A",
    )
    plot_depth_map_on_axes(
        ax_compared,
        fig,
        mark_compared.scan_image.data,
        mark_compared.scan_image.scale_x,
        title="Filtered Compared Surface B",
    )
    draw_metadata_box(
        ax_results, results_items, draw_border=False, wrap_width=wrap_width
    )

    # Row 2: side-by-side
    plot_side_by_side_on_axes(
        ax_side,
        fig,
        mark_reference_aligned.scan_image.data,
        mark_compared_aligned.scan_image.data,
        mark_reference.scan_image.scale_x,
    )

    # Row 3: profile plot
    plot_profiles_on_axes(
        ax_profile,
        profile_reference.heights,
        profile_compared.heights,
        profile_reference.pixel_size,
        metrics.correlation_coefficient,
        title="Reference Profile A / Moved Compared Profile B. Correlation Coefficient",
    )

    return finish_overview(
        fig,
        tight_layout_kwargs={"pad": 0.8, "h_pad": 1.2, "w_pad": 0.8},
        subplots_adjust_kwargs={
            "left": 0.06,
            "right": 0.98,
            "top": 0.96,
            "bottom": 0.06,
        },
    )


def build_striation_results_metadata(mark_reference: Mark, metrics: StriationComparisonResults) -> dict[
    str | Any, str | Any]:
    """Set up the overview of metadata to show in the plot."""
    results_items = {
        "Date report": datetime.now().strftime("%Y-%m-%d"),
        "Mark type": mark_reference.mark_type.value,
        "Correlation Coefficient": f"{metrics.correlation_coefficient:.4f}",
        "Overlap ratio": f"{metrics.overlap_ratio * 100:.2f} %",
        "Overlap length": f"{metrics.overlap_length * mega:.4f} µm",
        "Data spacing": f"{metrics.pixel_size * mega:.4f} µm",
        "Cutoff length low-pass filter": f"{val:.0f} µm"
        if (val := mark_reference.meta_data.get("lowpass_cutoff")) is not None
        else "N/A",
        "Cutoff length high-pass filter": f"{val:.0f} µm"
        if (val := mark_reference.meta_data.get("highpass_cutoff")) is not None
        else "N/A",
    }
    return results_items
