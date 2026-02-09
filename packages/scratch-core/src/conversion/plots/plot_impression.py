"""Impression mark comparison visualization."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from container_models.base import FloatArray2D, ImageRGB
from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    ImpressionComparisonMetrics,
    ImpressionComparisonPlots,
)
from conversion.plots.utils import (
    DEFAULT_COLORMAP,
    draw_metadata_box,
    figure_to_array,
    get_figure_dimensions,
    get_height_ratios,
    get_metadata_dimensions,
    plot_depth_map_on_axes,
    plot_depth_map_with_axes,
)


def plot_impression_comparison_results(
    mark_reference_leveled: Mark,
    mark_compared_leveled: Mark,
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    metrics: ImpressionComparisonMetrics,
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
) -> ImpressionComparisonPlots:
    """
    Generate visualization results for impression mark comparison.

    Main orchestrator function that generates both area-based and cell/CMC-based
    visualizations based on which results are available in the metrics.

    :param mark_reference_leveled: Reference mark after leveling.
    :param mark_compared_leveled: Compared mark after leveling.
    :param mark_reference_filtered: Reference mark after filtering.
    :param mark_compared_filtered: Compared mark after filtering.
    :param metrics: Comparison metrics including correlation values.
    :param metadata_reference: Metadata dict for reference mark display.
    :param metadata_compared: Metadata dict for compared mark display.
    :returns: ImpressionComparisonPlots with all rendered images.
    """
    # Initialize all plots as None
    leveled_ref = None
    leveled_comp = None
    filtered_ref = None
    filtered_comp = None
    cell_ref = None
    cell_comp = None
    cell_overlay = None
    cell_xcorr = None

    # Generate area-based plots if available
    if metrics.has_area_results:
        (
            leveled_ref,
            leveled_comp,
            filtered_ref,
            filtered_comp,
        ) = plot_area_figures(
            mark_ref_leveled=mark_reference_leveled,
            mark_comp_leveled=mark_compared_leveled,
            mark_ref_filtered=mark_reference_filtered,
            mark_comp_filtered=mark_compared_filtered,
        )

    # Generate cell/CMC-based plots if available
    if metrics.has_cell_results:
        (
            cell_ref,
            cell_comp,
            cell_overlay,
            cell_xcorr,
        ) = plot_cmc_figures(
            mark_ref_filtered=mark_reference_filtered,
            mark_comp_filtered=mark_compared_filtered,
            cell_correlations=metrics.cell_correlations,
        )

    # Generate comparison overview
    comparison_overview = plot_comparison_overview(
        mark_reference_leveled=mark_reference_leveled,
        mark_compared_leveled=mark_compared_leveled,
        mark_reference_filtered=mark_reference_filtered,
        mark_compared_filtered=mark_compared_filtered,
        metrics=metrics,
        metadata_reference=metadata_reference,
        metadata_compared=metadata_compared,
    )

    return ImpressionComparisonPlots(
        comparison_overview=comparison_overview,
        leveled_reference=leveled_ref,
        leveled_compared=leveled_comp,
        filtered_reference=filtered_ref,
        filtered_compared=filtered_comp,
        cell_reference=cell_ref,
        cell_compared=cell_comp,
        cell_overlay=cell_overlay,
        cell_cross_correlation=cell_xcorr,
    )


def plot_area_figures(
    mark_ref_leveled: Mark,
    mark_comp_leveled: Mark,
    mark_ref_filtered: Mark,
    mark_comp_filtered: Mark,
) -> tuple[ImageRGB, ImageRGB, ImageRGB, ImageRGB]:
    """
    Generate 4 area-based plots for impression comparison.

    Generates:
    1. Leveled reference surface
    2. Leveled compared surface
    3. Filtered reference surface
    4. Filtered compared surface

    :param mark_ref_leveled: Reference mark after leveling.
    :param mark_comp_leveled: Compared mark after leveling.
    :param mark_ref_filtered: Reference mark after filtering.
    :param mark_comp_filtered: Compared mark after filtering.
    :returns: Tuple of 4 ImageRGB arrays.
    """
    scale_ref = mark_ref_leveled.scan_image.scale_x
    scale_comp = mark_comp_leveled.scan_image.scale_x

    # 1. Leveled reference surface
    leveled_ref = plot_depth_map_with_axes(
        data=mark_ref_leveled.scan_image.data,
        scale=scale_ref,
        title="Leveled Reference Surface",
    )

    # 2. Leveled compared surface
    leveled_comp = plot_depth_map_with_axes(
        data=mark_comp_leveled.scan_image.data,
        scale=scale_comp,
        title="Leveled Compared Surface",
    )

    # 3. Filtered reference surface
    filtered_ref = plot_depth_map_with_axes(
        data=mark_ref_filtered.scan_image.data,
        scale=scale_ref,
        title="Filtered Reference Surface",
    )

    # 4. Filtered compared surface
    filtered_comp = plot_depth_map_with_axes(
        data=mark_comp_filtered.scan_image.data,
        scale=scale_comp,
        title="Filtered Compared Surface",
    )

    return leveled_ref, leveled_comp, filtered_ref, filtered_comp


def plot_cmc_figures(
    mark_ref_filtered: Mark,
    mark_comp_filtered: Mark,
    cell_correlations: FloatArray2D,
) -> tuple[ImageRGB, ImageRGB, ImageRGB, ImageRGB]:
    """
    Generate 4 CMC/cell-based plots for impression comparison.

    Generates:
    1. Cell-preprocessed reference
    2. Cell-preprocessed compared
    3. All cells overlay visualization
    4. Cell cross-correlation heatmap

    :param mark_ref_filtered: Reference mark after filtering.
    :param mark_comp_filtered: Compared mark after filtering.
    :param cell_correlations: Grid of per-cell correlation values.
    :returns: Tuple of 4 ImageRGB arrays.
    """
    scale = mark_ref_filtered.scan_image.scale_x

    # 1. Cell-preprocessed reference
    cell_ref = plot_depth_map_with_axes(
        data=mark_ref_filtered.scan_image.data,
        scale=scale,
        title="Cell-Preprocessed Reference",
    )

    # 2. Cell-preprocessed compared
    cell_comp = plot_depth_map_with_axes(
        data=mark_comp_filtered.scan_image.data,
        scale=scale,
        title="Cell-Preprocessed Compared",
    )

    # 3. Cell overlay visualization
    cell_overlay = plot_cell_grid_overlay(
        data=mark_ref_filtered.scan_image.data,
        scale=scale,
        cell_correlations=cell_correlations,
    )

    # 4. Cell cross-correlation heatmap
    cell_xcorr = plot_cell_correlation_heatmap(
        cell_correlations=cell_correlations,
    )

    return cell_ref, cell_comp, cell_overlay, cell_xcorr


def plot_comparison_overview(
    mark_reference_leveled: Mark,
    mark_compared_leveled: Mark,
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    metrics: ImpressionComparisonMetrics,
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
    wrap_width: int = 25,
) -> ImageRGB:
    """
    Generate the main results overview figure with dynamic sizing.

    Combines metadata tables, surface visualizations, cell grid overlay,
    and cell correlation heatmap into a single overview figure.

    :param mark_reference_leveled: Reference mark after leveling.
    :param mark_compared_leveled: Compared mark after leveling.
    :param mark_reference_filtered: Reference mark after filtering.
    :param mark_compared_filtered: Compared mark after filtering.
    :param metrics: Comparison metrics including correlation values.
    :param metadata_reference: Metadata dict for reference mark display.
    :param metadata_compared: Metadata dict for compared mark display.
    :param wrap_width: Maximum characters per line before wrapping.
    :returns: RGB image as uint8 array.
    """
    # Build results metadata
    results_items = {
        "Date report": datetime.now().strftime("%Y-%m-%d"),
        "Mark type": mark_reference_leveled.mark_type.value,
        "Area Correlation": f"{metrics.area_correlation:.4f}",
        "CMC Score": f"{metrics.cmc_score:.1f}%",
        "Sq(Ref)": f"{metrics.sq_ref:.4f} µm",
        "Sq(Comp)": f"{metrics.sq_comp:.4f} µm",
        "Sq(Diff)": f"{metrics.sq_diff:.4f} µm",
    }

    max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
        metadata_compared, metadata_reference, wrap_width
    )
    height_ratios = get_height_ratios(metadata_height_ratio)

    # Adjust figure height based on content
    fig_height = 14 + (max_metadata_rows * 0.12)
    fig_height = max(13, min(17, fig_height))

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
    draw_metadata_box(
        ax_meta_reference,
        metadata_reference,
        "Reference Mark (A)",
        wrap_width=wrap_width,
    )

    ax_meta_compared = fig.add_subplot(gs[0, 1])
    draw_metadata_box(
        ax_meta_compared,
        metadata_compared,
        "Compared Mark (B)",
        wrap_width=wrap_width,
    )

    # Row 1: Leveled surfaces + Results
    ax_leveled_ref = fig.add_subplot(gs[1, 0])
    plot_depth_map_on_axes(
        ax_leveled_ref,
        fig,
        mark_reference_leveled.scan_image.data,
        mark_reference_leveled.scan_image.scale_x,
        title="Leveled Reference Surface A",
    )

    ax_leveled_comp = fig.add_subplot(gs[1, 1])
    plot_depth_map_on_axes(
        ax_leveled_comp,
        fig,
        mark_compared_leveled.scan_image.data,
        mark_compared_leveled.scan_image.scale_x,
        title="Leveled Compared Surface B",
    )

    ax_results = fig.add_subplot(gs[1, 2])
    draw_metadata_box(
        ax_results, results_items, draw_border=False, wrap_width=wrap_width
    )

    # Row 2: Filtered surfaces
    ax_filtered_ref = fig.add_subplot(gs[2, 0])
    plot_depth_map_on_axes(
        ax_filtered_ref,
        fig,
        mark_reference_filtered.scan_image.data,
        mark_reference_filtered.scan_image.scale_x,
        title="Filtered Reference Surface A",
    )

    ax_filtered_comp = fig.add_subplot(gs[2, 1])
    plot_depth_map_on_axes(
        ax_filtered_comp,
        fig,
        mark_compared_filtered.scan_image.data,
        mark_compared_filtered.scan_image.scale_x,
        title="Filtered Compared Surface B",
    )

    # Row 2, Col 2: Cell correlation heatmap (if available)
    if metrics.has_cell_results:
        ax_heatmap = fig.add_subplot(gs[2, 2])
        _plot_cell_heatmap_on_axes(ax_heatmap, fig, metrics.cell_correlations)

    # Row 3: Cell grid overlay (spanning full width if cell results available)
    if metrics.has_cell_results:
        ax_overlay = fig.add_subplot(gs[3, :2])
        _plot_cell_overlay_on_axes(
            ax_overlay,
            mark_reference_filtered.scan_image.data,
            mark_reference_filtered.scan_image.scale_x,
            metrics.cell_correlations,
        )

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.8)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.06)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_cell_grid_overlay(
    data: FloatArray2D,
    scale: float,
    cell_correlations: FloatArray2D,
) -> ImageRGB:
    """
    Plot surface with cell grid overlay showing correlation values.

    :param data: Surface data in meters.
    :param scale: Pixel scale in meters.
    :param cell_correlations: Grid of per-cell correlation values.
    :returns: RGB image as uint8 array.
    """
    height, width = data.shape
    fig_height, fig_width = get_figure_dimensions(height, width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    _plot_cell_overlay_on_axes(ax, data, scale, cell_correlations)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_cell_correlation_heatmap(
    cell_correlations: FloatArray2D,
) -> ImageRGB:
    """
    Plot heatmap of per-cell correlation values.

    :param cell_correlations: Grid of per-cell correlation values.
    :returns: RGB image as uint8 array.
    """
    n_rows, n_cols = cell_correlations.shape

    # Calculate figure size based on grid dimensions
    base_size = 6
    aspect = n_cols / n_rows
    if aspect > 1:
        fig_width = base_size
        fig_height = base_size / aspect + 1.5
    else:
        fig_height = base_size + 1.5
        fig_width = base_size * aspect

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    _plot_cell_heatmap_on_axes(ax, fig, cell_correlations)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


# --- Helper functions for axes-level plotting ---


def _plot_cell_overlay_on_axes(
    ax: Axes,
    data: FloatArray2D,
    scale: float,
    cell_correlations: FloatArray2D,
) -> None:
    """Plot surface with cell grid overlay on given axes."""
    height, width = data.shape
    n_rows, n_cols = cell_correlations.shape

    # Plot the surface
    extent = (0, width * scale * 1e6, 0, height * scale * 1e6)
    ax.imshow(
        data * 1e6,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin="lower",
        extent=extent,
    )

    # Calculate cell dimensions
    cell_height = height / n_rows
    cell_width = width / n_cols

    # Draw grid and correlation values
    for i in range(n_rows):
        for j in range(n_cols):
            # Cell boundaries in um
            x_left = j * cell_width * scale * 1e6
            x_right = (j + 1) * cell_width * scale * 1e6
            y_bottom = (n_rows - 1 - i) * cell_height * scale * 1e6
            y_top = (n_rows - i) * cell_height * scale * 1e6

            # Draw cell border
            ax.plot(
                [x_left, x_right, x_right, x_left, x_left],
                [y_bottom, y_bottom, y_top, y_top, y_bottom],
                "w-",
                linewidth=0.5,
                alpha=0.7,
            )

            # Add correlation value text
            corr_val = cell_correlations[i, j]
            if not np.isnan(corr_val):
                x_center = (x_left + x_right) / 2
                y_center = (y_bottom + y_top) / 2
                ax.text(
                    x_center,
                    y_center,
                    f"{corr_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
                )

    ax.set_xlabel("X - Position [µm]", fontsize=11)
    ax.set_ylabel("Y - Position [µm]", fontsize=11)
    ax.set_title("Cell Grid with Correlation Values", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)


def _plot_cell_heatmap_on_axes(
    ax: Axes,
    fig: Figure,
    cell_correlations: FloatArray2D,
) -> None:
    """Plot cell correlation heatmap on given axes."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_rows, n_cols = cell_correlations.shape

    im = ax.imshow(
        cell_correlations,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    # Add cell value annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = cell_correlations[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )

    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Row", fontsize=11)
    ax.set_title("Cell Correlation Heatmap", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)

    # Set tick positions
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label="Correlation")
    cbar.ax.tick_params(labelsize=10)
