from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from container_models.base import FloatArray2D, ImageRGB
from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    ImpressionComparisonMetrics,
    ImpressionComparisonPlots,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from conversion.plots.utils import (
    DEFAULT_COLORMAP,
    draw_metadata_box,
    figure_to_array,
    get_figure_dimensions,
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

    :param mark_reference_leveled: Reference mark after leveling.
    :param mark_compared_leveled: Compared mark after leveling.
    :param mark_reference_filtered: Reference mark after filtering.
    :param mark_compared_filtered: Compared mark after filtering.
    :param metrics: Comparison metrics including correlation values.
    :param metadata_reference: Metadata dict for reference mark display.
    :param metadata_compared: Metadata dict for compared mark display.
    :returns: ImpressionComparisonPlots with all rendered images.
    """
    # Area-based plots (leveled + filtered surfaces)
    leveled_ref = plot_depth_map_with_axes(
        data=mark_reference_leveled.scan_image.data,
        scale=mark_reference_leveled.scan_image.scale_x,
        title="Leveled Reference Surface",
    )
    leveled_comp = plot_depth_map_with_axes(
        data=mark_compared_leveled.scan_image.data,
        scale=mark_compared_leveled.scan_image.scale_x,
        title="Leveled Compared Surface",
    )
    filtered_ref = plot_depth_map_with_axes(
        data=mark_reference_filtered.scan_image.data,
        scale=mark_reference_leveled.scan_image.scale_x,
        title="Filtered Reference Surface",
    )
    filtered_comp = plot_depth_map_with_axes(
        data=mark_compared_filtered.scan_image.data,
        scale=mark_compared_leveled.scan_image.scale_x,
        title="Filtered Compared Surface",
    )

    # Cell/CMC-based plots
    scale = mark_reference_filtered.scan_image.scale_x
    cell_ref = plot_depth_map_with_axes(
        data=mark_reference_filtered.scan_image.data,
        scale=scale,
        title="Cell-Preprocessed Reference",
    )
    cell_comp = plot_depth_map_with_axes(
        data=mark_compared_filtered.scan_image.data,
        scale=scale,
        title="Cell-Preprocessed Compared",
    )
    cell_overlay = plot_cell_grid_overlay(
        data=mark_reference_filtered.scan_image.data,
        scale=scale,
        cell_correlations=metrics.cell_correlations,
        cell_similarity_threshold=metrics.cell_similarity_threshold,
    )
    cell_xcorr = plot_cell_correlation_heatmap(
        cell_correlations=metrics.cell_correlations,
    )

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
        leveled_reference_surface_map=leveled_ref,
        leveled_compared_surface_map=leveled_comp,
        filtered_reference_surface_map=filtered_ref,
        filtered_compared_surface_map=filtered_comp,
        cell_reference_surface_map=cell_ref,
        cell_compared_surface_map=cell_comp,
        cell_overlay=cell_overlay,
        cell_cross_correlation=cell_xcorr,
    )


def plot_cell_grid_overlay(
    data: FloatArray2D,
    scale: float,
    cell_correlations: FloatArray2D,
    cell_label_prefix: str = "A",
    cell_similarity_threshold: float = 0.25,
    show_all_cells: bool = True,
    cell_positions: np.ndarray | None = None,
    cell_rotations: np.ndarray | None = None,
    cell_size_um: tuple[float, float] | None = None,
) -> ImageRGB:
    """
    Plot surface with cell grid overlay showing cell names and CMC status.

    Cells above the similarity threshold are drawn with black outlines,
    cells below the threshold with red outlines.

    :param data: Surface data in meters.
    :param scale: Pixel scale in meters.
    :param cell_correlations: Grid of per-cell correlation values.
    :param cell_label_prefix: Label prefix for cells ("A" for reference, "B" for compared).
    :param cell_similarity_threshold: Minimum correlation for a cell to be considered CMC.
    :param show_all_cells: If True, show all cells. If False, only show CMC cells.
    :param cell_positions: (n_cells, 2) array of (x, y) positions in µm, row-major order.
    :param cell_rotations: (n_cells,) array of rotation angles in radians, row-major order.
    :param cell_size_um: (width, height) of a cell in µm (required when cell_positions is set).
    :returns: RGB image as uint8 array.
    """
    height, width = data.shape
    fig_height, fig_width = get_figure_dimensions(height, width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    _plot_cell_overlay_on_axes(
        ax,
        data,
        scale,
        cell_correlations,
        cell_label_prefix=cell_label_prefix,
        cell_similarity_threshold=cell_similarity_threshold,
        show_all_cells=show_all_cells,
        cell_positions=cell_positions,
        cell_rotations=cell_rotations,
        cell_size_um=cell_size_um,
    )

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

    Combines metadata tables, surface visualizations, and cell correlation
    heatmap into a single overview figure.

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
    # Build results metadata matching MATLAB nfi_nist_plot_cmc style
    n_cell_rows, n_cell_cols = metrics.cell_correlations.shape
    n_cells = n_cell_rows * n_cell_cols
    n_cmc = int(np.sum(metrics.cell_correlations >= metrics.cell_similarity_threshold))
    cmc_fraction = n_cmc / n_cells * 100 if n_cells > 0 else 0.0

    scale_x_um = mark_reference_filtered.scan_image.scale_x * 1e6
    scale_y_um = mark_reference_filtered.scan_image.scale_y * 1e6

    results_items: dict[str, str] = {
        "Date report": datetime.now().strftime("%Y-%m-%d"),
        "Mark type": mark_reference_leveled.mark_type.value,
        "Number of Cells": str(n_cells),
        "Number of CMCs": str(n_cmc),
        "CMC fraction": f"{cmc_fraction:.2f} %",
    }
    results_items["CMC area fraction"] = f"{metrics.cmc_area_fraction:.2f} %"
    results_items["Data spacing (X)"] = f"{scale_x_um:.4f} µm"
    results_items["Data spacing (Y)"] = f"{scale_y_um:.4f} µm"
    results_items["Cutoff length low-pass filter"] = f"{metrics.cutoff_low_pass:.0f} µm"
    results_items["Cutoff length high-pass filter"] = (
        f"{metrics.cutoff_high_pass:.0f} µm"
    )
    results_items["Cell Size"] = f"{metrics.cell_size_um:.0f} µm"
    results_items["Minimum cell similarity"] = f"{metrics.cell_similarity_threshold}"
    results_items["Max error cell position"] = (
        f"{metrics.max_error_cell_position:.0f} µm"
    )
    results_items["Max error cell angle"] = f"{metrics.max_error_cell_angle:.0f} degree"

    max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
        metadata_compared, metadata_reference, wrap_width
    )

    # 3-row layout: metadata, leveled surfaces + results, filtered surfaces + heatmap
    row1_height = 0.40
    row2_height = 0.40
    total = metadata_height_ratio + row1_height + row2_height
    height_ratios = [
        metadata_height_ratio / total,
        row1_height / total,
        row2_height / total,
    ]

    # Adjust figure height based on content
    fig_height = 12 + (max_metadata_rows * 0.12)
    fig_height = max(10.0, min(15.0, fig_height))

    fig = plt.figure(figsize=(16, fig_height))

    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=height_ratios,
        width_ratios=[0.35, 0.35, 0.30],
        hspace=0.35,
        wspace=0.45,
    )

    # Row 0: Metadata tables — span full width as two equal columns
    gs_meta = gs[0, :].subgridspec(1, 2, wspace=0.15)

    ax_meta_reference = fig.add_subplot(gs_meta[0, 0])
    draw_metadata_box(
        ax_meta_reference,
        metadata_reference,
        "Reference Surface (A)",
        wrap_width=wrap_width,
    )

    ax_meta_compared = fig.add_subplot(gs_meta[0, 1])
    draw_metadata_box(
        ax_meta_compared,
        metadata_compared,
        "Compared Surface (B)",
        wrap_width=wrap_width,
    )

    # Row 1: Leveled surfaces + Results
    ax_leveled_ref = fig.add_subplot(gs[1, 0])
    plot_depth_map_on_axes(
        ax_leveled_ref,
        fig,
        mark_reference_leveled.scan_image.data,
        mark_reference_leveled.scan_image.scale_x,
        title="Reference Surface A",
    )

    ax_leveled_comp = fig.add_subplot(gs[1, 1])
    plot_depth_map_on_axes(
        ax_leveled_comp,
        fig,
        mark_compared_leveled.scan_image.data,
        mark_compared_leveled.scan_image.scale_x,
        title="Compared Surface B",
    )

    ax_results = fig.add_subplot(gs[1, 2])
    draw_metadata_box(
        ax_results,
        results_items,
        draw_border=False,
        wrap_width=wrap_width,
    )

    # Row 2: Filtered surfaces with cell grid overlay + Cell ACCF Distribution
    ax_filtered_ref = fig.add_subplot(gs[2, 0])
    im_ref = _plot_cell_overlay_on_axes(
        ax_filtered_ref,
        mark_reference_filtered.scan_image.data,
        mark_reference_filtered.scan_image.scale_x,
        metrics.cell_correlations,
        cell_label_prefix="A",
        cell_similarity_threshold=metrics.cell_similarity_threshold,
        show_all_cells=True,
    )
    ax_filtered_ref.set_title(
        "Filtered Reference Surface A", fontsize=12, fontweight="bold"
    )
    divider_ref = make_axes_locatable(ax_filtered_ref)
    cax_ref = divider_ref.append_axes("right", size="5%", pad=0.05)
    cbar_ref = fig.colorbar(im_ref, cax=cax_ref, label="Scan Depth [µm]")
    cbar_ref.ax.tick_params(labelsize=9)

    ax_filtered_comp = fig.add_subplot(gs[2, 1])
    ref_h, ref_w = mark_reference_filtered.scan_image.data.shape
    ref_scale = mark_reference_filtered.scan_image.scale_x
    n_rows, n_cols = metrics.cell_correlations.shape
    cell_size_um = (
        ref_w * ref_scale * 1e6 / n_cols,
        ref_h * ref_scale * 1e6 / n_rows,
    )

    im_comp = _plot_cell_overlay_on_axes(
        ax_filtered_comp,
        mark_compared_filtered.scan_image.data,
        mark_compared_filtered.scan_image.scale_x,
        metrics.cell_correlations,
        cell_label_prefix="B",
        cell_similarity_threshold=metrics.cell_similarity_threshold,
        show_all_cells=False,
        cell_positions=metrics.cell_positions_compared,
        cell_rotations=metrics.cell_rotations_compared,
        cell_size_um=cell_size_um
        if metrics.cell_positions_compared is not None
        else None,
    )
    ax_filtered_comp.set_title(
        "Filtered Compared Surface B", fontsize=12, fontweight="bold"
    )
    divider_comp = make_axes_locatable(ax_filtered_comp)
    cax_comp = divider_comp.append_axes("right", size="5%", pad=0.05)
    cbar_comp = fig.colorbar(im_comp, cax=cax_comp, label="Scan Depth [µm]")
    cbar_comp.ax.tick_params(labelsize=9)

    ax_heatmap = fig.add_subplot(gs[2, 2])
    ref_data = mark_reference_filtered.scan_image.data
    ref_sc = mark_reference_filtered.scan_image.scale_x
    heatmap_extent_um = (
        ref_data.shape[1] * ref_sc * 1e6,
        ref_data.shape[0] * ref_sc * 1e6,
    )
    _plot_cell_heatmap_on_axes(
        ax_heatmap,
        fig,
        metrics.cell_correlations,
        surface_extent_um=heatmap_extent_um,
        cell_label_prefix="A",
        cell_similarity_threshold=metrics.cell_similarity_threshold,
    )

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.8)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.96, bottom=0.06)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def _plot_cell_overlay_on_axes(
    ax: Axes,
    data: FloatArray2D,
    scale: float,
    cell_correlations: FloatArray2D,
    cell_label_prefix: str = "A",
    cell_similarity_threshold: float = 0.25,
    show_all_cells: bool = True,
    cell_positions: np.ndarray | None = None,
    cell_rotations: np.ndarray | None = None,
    cell_size_um: tuple[float, float] | None = None,
) -> AxesImage:
    """
    Plot surface with cell grid overlay on given axes.

    Follows the MATLAB plot_cells convention: cells above the similarity
    threshold (CMC cells) are drawn with black outlines and labels, while
    cells below the threshold are drawn with red outlines and labels.

    When ``cell_positions`` is provided, each cell is drawn as a (possibly
    rotated) rectangle at the given position instead of at a regular grid
    location. This is used for the compared surface where cells appear at
    their matched positions (``Cell(i).vPos2`` / ``Cell(i).angle2`` in
    MATLAB).

    :param ax: Matplotlib axes to plot on.
    :param data: Surface data in meters.
    :param scale: Pixel scale in meters.
    :param cell_correlations: Grid of per-cell correlation values.
    :param cell_label_prefix: Label prefix for cells ("A" for reference, "B" for compared).
    :param cell_similarity_threshold: Minimum correlation for a cell to be considered CMC.
    :param show_all_cells: If True, show all cells. If False, only show CMC cells.
    :param cell_positions: (n_cells, 2) array of (x, y) positions in µm, row-major order.
    :param cell_rotations: (n_cells,) array of rotation angles in radians, row-major order.
    :param cell_size_um: (width, height) of a cell in µm (required when cell_positions is set).
    """
    height, width = data.shape
    n_rows, n_cols = cell_correlations.shape

    # Plot the surface
    extent = (0, width * scale * 1e6, 0, height * scale * 1e6)
    im = ax.imshow(
        data * 1e6,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin="lower",
        extent=extent,
    )

    # Calculate cell dimensions in pixels (for grid-based drawing)
    cell_height = height / n_rows
    cell_width = width / n_cols

    # Collect cells into CMC (black) and non-CMC (red) groups.
    # Draw CMC cells first, then non-CMC on top so red outlines are not
    # hidden by adjacent black cell borders.
    cmc_cells: list[tuple[int, int, int]] = []
    non_cmc_cells: list[tuple[int, int, int]] = []

    cell_index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            cell_index += 1
            corr_val = cell_correlations[i, j]

            if np.isnan(corr_val):
                continue

            is_cmc = corr_val >= cell_similarity_threshold

            if not is_cmc and not show_all_cells:
                continue

            if is_cmc:
                cmc_cells.append((i, j, cell_index))
            else:
                non_cmc_cells.append((i, j, cell_index))

    use_custom_positions = cell_positions is not None and cell_size_um is not None

    for color, cells in [("black", cmc_cells), ("red", non_cmc_cells)]:
        for i, j, idx in cells:
            flat_index = i * n_cols + j  # row-major flat index

            if use_custom_positions:
                assert cell_positions is not None
                assert cell_size_um is not None
                cx = float(cell_positions[flat_index, 0])
                cy = float(cell_positions[flat_index, 1])
                w, h = cell_size_um

                if np.isnan(cx) or np.isnan(cy):
                    continue

                # Build rectangle corners centered at origin
                half_w, half_h = w / 2, h / 2
                corners = np.array(
                    [
                        [-half_w, -half_h],
                        [half_w, -half_h],
                        [half_w, half_h],
                        [-half_w, half_h],
                    ]
                )

                # Rotate
                angle = (
                    float(cell_rotations[flat_index])
                    if cell_rotations is not None
                    else 0.0
                )
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                corners = corners @ rot.T

                # Translate to position
                corners[:, 0] += cx
                corners[:, 1] += cy

                # Draw closed polygon
                xs = np.append(corners[:, 0], corners[0, 0])
                ys = np.append(corners[:, 1], corners[0, 1])
                ax.plot(xs, ys, color=color, linestyle="-", linewidth=1.0)

                # Label at center
                ax.text(
                    cx,
                    cy,
                    f"{cell_label_prefix}{idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )
            else:
                # Grid-based cell boundaries in µm
                x_left = j * cell_width * scale * 1e6
                x_right = (j + 1) * cell_width * scale * 1e6
                y_bottom = (n_rows - 1 - i) * cell_height * scale * 1e6
                y_top = (n_rows - i) * cell_height * scale * 1e6

                # Draw cell border
                ax.plot(
                    [x_left, x_right, x_right, x_left, x_left],
                    [y_bottom, y_bottom, y_top, y_top, y_bottom],
                    color=color,
                    linestyle="-",
                    linewidth=1.0,
                )

                # Add cell name label
                x_center = (x_left + x_right) / 2
                y_center = (y_bottom + y_top) / 2
                ax.text(
                    x_center,
                    y_center,
                    f"{cell_label_prefix}{idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )

    ax.set_xlabel("X - Position [µm]", fontsize=11)
    ax.set_ylabel("Y - Position [µm]", fontsize=11)
    ax.tick_params(labelsize=10)

    return im


def _plot_cell_heatmap_on_axes(
    ax: Axes,
    fig: Figure,
    cell_correlations: FloatArray2D,
    surface_extent_um: tuple[float, float] | None = None,
    cell_label_prefix: str = "A",
    cell_similarity_threshold: float = 0.25,
) -> None:
    """
    Plot cell correlation heatmap on given axes.

    When ``surface_extent_um`` is provided the heatmap uses spatial µm axes
    and shows cell labels (A1, A2, …) coloured by CMC status, matching the
    MATLAB ``nfi_nist_plot_cmc`` style.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param cell_correlations: Grid of per-cell correlation values.
    :param surface_extent_um: (width, height) of the surface in µm.
    :param cell_label_prefix: Prefix for cell labels.
    :param cell_similarity_threshold: Minimum correlation for CMC.
    """
    n_rows, n_cols = cell_correlations.shape

    if surface_extent_um is not None:
        w_um, h_um = surface_extent_um
        extent = (0, w_um, 0, h_um)
        origin = "lower"
    else:
        extent = None
        origin = "upper"

    im = ax.imshow(
        cell_correlations,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin=origin,
        extent=extent,
        vmin=0,
        vmax=1,
    )

    # Add cell labels
    cell_w = (surface_extent_um[0] / n_cols) if surface_extent_um else 1.0
    cell_h = (surface_extent_um[1] / n_rows) if surface_extent_um else 1.0

    cell_index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            cell_index += 1
            val = cell_correlations[i, j]
            if np.isnan(val):
                continue

            if surface_extent_um is not None:
                cx = (j + 0.5) * cell_w
                cy = (n_rows - 1 - i + 0.5) * cell_h
                is_cmc = val >= cell_similarity_threshold
                color = "blue" if is_cmc else "red"
                label = f"{cell_label_prefix}{cell_index}"
            else:
                cx = j
                cy = i
                color = "white" if val < 0.5 else "black"
                label = f"{val:.2f}"

            ax.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.set_title("Cell ACCF Distribution", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)

    if surface_extent_um is not None:
        ax.set_xlabel("X - Position [µm]", fontsize=11)
        ax.set_ylabel("Y - Position [µm]", fontsize=11)
    else:
        ax.set_xlabel("Column", fontsize=11)
        ax.set_ylabel("Row", fontsize=11)
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
