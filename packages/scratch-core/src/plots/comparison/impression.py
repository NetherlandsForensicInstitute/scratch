from datetime import datetime
from typing import Sequence, Any, Literal

import matplotlib.pyplot as plt
from scipy.constants import mega
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from container_models.base import FloatArray2D, ImageRGB
from conversion.data_formats import Mark, MarkMetadata
from plots.cell_overlays import (
    plot_cell_overlay_on_axes,
    _plot_cell_heatmap_on_axes,
)
from plots.comparison.data_formats import ImpressionComparisonPlots

from plots.utils import (
    finish_overview,
    render_single_panel,
    get_figure_dimensions,
    get_height_ratios,
    overview_figure_height,
)
from plots.on_axes import (
    plot_depth_map_on_axes,
    plot_depth_map_with_axes,
    _plot_surface_with_colorbar,
)
from plots.metadata_tables import (
    get_metadata_dimensions,
    draw_metadata_box,
    draw_metadata_pair,
)

from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)
from conversion.surface_comparison.utils import _cells_correlation_to_grid


def plot_impression_comparison_results(
    mark_reference_raw: Mark,
    mark_compared_raw: Mark,
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    cmc_result: ComparisonResult,
    comparison_params: ComparisonParams,
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
) -> ImpressionComparisonPlots:
    """
    Generate visualization results for impression mark comparison.

    :param mark_reference_raw: Raw reference mark.
    :param mark_compared_raw: Raw compared mark.
    :param mark_reference_filtered: Reference mark after filtering.
    :param mark_compared_filtered: Compared mark after filtering.
    :param cmc_result: Result of the CMC algorithm
    :param comparison_params: Parameters for the CMC algorithm.
    :param metadata_reference: Metadata dict for reference mark display.
    :param metadata_compared: Metadata dict for compared mark display.
    :returns: ImpressionComparisonPlots with all rendered images.
    """
    # Area-based plots (raw + filtered surfaces)
    raw_ref = plot_depth_map_with_axes(
        data=mark_reference_raw.scan_image.data,
        scale=mark_reference_raw.scan_image.scale_x,
        title="Raw Reference Surface",
    )
    raw_comp = plot_depth_map_with_axes(
        data=mark_compared_raw.scan_image.data,
        scale=mark_compared_raw.scan_image.scale_x,
        title="Raw Compared Surface",
    )
    filtered_ref = plot_depth_map_with_axes(
        data=mark_reference_filtered.scan_image.data,
        scale=mark_reference_filtered.scan_image.scale_x,
        title="Filtered Reference Surface",
    )
    filtered_comp = plot_depth_map_with_axes(
        data=mark_compared_filtered.scan_image.data,
        scale=mark_compared_filtered.scan_image.scale_x,
        title="Filtered Compared Surface",
    )

    # Cell/CMC-based plots
    scale = mark_reference_filtered.scan_image.scale_x
    cell_ref = plot_cell_grid_overlay(
        data=mark_reference_filtered.scan_image.data,
        scale=scale,
        cells=cmc_result.cells,
        cell_label_prefix="A",
        space="reference",
        show_all_cells=True,
        title="Reference Surface A — All Cells",
    )
    cell_comp = plot_cell_grid_overlay(
        data=mark_compared_filtered.scan_image.data,
        scale=mark_compared_filtered.scan_image.scale_x,
        cells=cmc_result.cells,
        cell_label_prefix="B",
        show_all_cells=False,
        space="comparison",
        title="Compared Surface B — CMC Cells",
    )
    cell_overlay = plot_cell_grid_overlay(
        data=mark_compared_filtered.scan_image.data,
        scale=mark_compared_filtered.scan_image.scale_x,
        cells=cmc_result.cells,
        title="Compared Surface B — All Cells",
        cell_label_prefix="B",
        show_all_cells=True,
        space="comparison",
    )
    ref_data = mark_reference_filtered.scan_image.data
    surface_extent_um = (
        ref_data.shape[1] * scale * mega,
        ref_data.shape[0] * scale * mega,
    )
    cell_correlation = plot_cell_correlation_heatmap(
        cells=cmc_result.cells,
        surface_extent_um=surface_extent_um,
    )

    comparison_overview = plot_impression_comparison_overview(
        mark_reference_raw=mark_reference_raw,
        mark_compared_raw=mark_compared_raw,
        mark_reference_filtered=mark_reference_filtered,
        mark_compared_filtered=mark_compared_filtered,
        cmc_result=cmc_result,
        comparison_params=comparison_params,
        metadata_reference=metadata_reference,
        metadata_compared=metadata_compared,
    )

    return ImpressionComparisonPlots(
        comparison_overview=comparison_overview,
        raw_reference_heatmap=raw_ref,
        raw_compared_heatmap=raw_comp,
        filtered_reference_heatmap=filtered_ref,
        filtered_compared_heatmap=filtered_comp,
        cell_reference_heatmap=cell_ref,
        cell_compared_heatmap=cell_comp,
        cell_overlay=cell_overlay,
        cell_cross_correlation=cell_correlation,
    )


def plot_cell_grid_overlay(
    data: FloatArray2D,
    scale: float,
    cells: Sequence[Cell],
    space: Literal["reference", "comparison"],
    title: str = "Cell Grid Overlay",
    cell_label_prefix: str = "A",
    show_all_cells: bool = True,
) -> ImageRGB:
    """
    Plot surface with cell grid overlay showing cell names and CMC status.

    Cells above the similarity threshold are drawn with black outlines,
    cells below the threshold with red outlines.

    :param data: Surface data in meters.
    :param scale: Pixel scale in meters.
    :param cells: Cells to plot.
    :param space: which space to plot the cells on
    :param title: title of the plot
    :param cell_label_prefix: Label prefix for cells ("A" for reference, "B" for compared).
    :param show_all_cells: If True, show all cells. If False, only show CMC cells.
    :returns: RGB image as uint8 array.
    """
    height, width = data.shape
    fig_height, fig_width = get_figure_dimensions(height, width)

    def draw(fig: Figure, ax: Axes) -> None:
        im = plot_cell_overlay_on_axes(
            ax,
            data,
            scale,
            cells=cells,
            cell_label_prefix=cell_label_prefix,
            show_all_cells=show_all_cells,
            space=space,
        )
        _plot_surface_with_colorbar(fig, ax, im, title=title)

    return render_single_panel((fig_width, fig_height), draw)


def plot_cell_correlation_heatmap(
    cells: Sequence[Cell],
    surface_extent_um: tuple[float, float],
) -> ImageRGB:
    """
    Plot heatmap of per-cell correlation values.

    :param cells: Cell results from the CMC pipeline.
    :param surface_extent_um: (width, height) of the surface in µm.
    :returns: RGB image as uint8 array.
    """
    cell_correlations = _cells_correlation_to_grid(cells)
    n_rows, n_cols = cell_correlations.shape

    base_size = 6
    aspect = n_cols / n_rows
    if aspect > 1:
        fig_width = base_size
        fig_height = base_size / aspect + 1.5
    else:
        fig_height = base_size + 1.5
        fig_width = base_size * aspect

    return render_single_panel(
        (fig_width, fig_height),
        lambda fig, ax: _plot_cell_heatmap_on_axes(
            ax,
            fig,
            cells=cells,
            cell_correlations=cell_correlations,
            surface_extent_um=surface_extent_um,
        ),
    )


def plot_impression_comparison_overview(
    mark_reference_raw: Mark,
    mark_compared_raw: Mark,
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    cmc_result: ComparisonResult,
    comparison_params: ComparisonParams,
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
    wrap_width: int = 25,
) -> ImageRGB:
    """
    Generate the main results overview figure with dynamic sizing.

    Combines metadata tables, surface visualizations, and cell correlation
    heatmap into a single overview figure.

    :param mark_reference_raw: Raw reference mark.
    :param mark_compared_raw: Raw compared mark.
    :param mark_reference_filtered: Reference mark after filtering.
    :param mark_compared_filtered: Compared mark after filtering.
    :param cmc_result: Result of the CMC algorithm
    :param comparison_params: Parameters for the CMC algorithm.
    :param metadata_reference: Metadata object for reference mark display.
    :param metadata_compared: Metadata object for compared mark display.
    :param wrap_width: Maximum characters per line before wrapping.
    :returns: RGB image as uint8 array.
    """
    cells = cmc_result.cells
    cell_correlations = _cells_correlation_to_grid(cells)

    scale_x_um = mark_reference_filtered.scan_image.scale_x * mega
    scale_y_um = mark_reference_filtered.scan_image.scale_y * mega

    results_items = build_impression_results_metadata(
        cells, cmc_result, comparison_params, mark_reference_raw, scale_x_um, scale_y_um
    )

    max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
        metadata_compared, metadata_reference, wrap_width
    )

    height_ratios = get_height_ratios(metadata_height_ratio, 0.40, 0.40)

    fig_height = overview_figure_height(max_metadata_rows, 12, 10.0, 15.0)

    fig = plt.figure(figsize=(16, fig_height))

    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=height_ratios,
        width_ratios=[0.35, 0.35, 0.30],
        hspace=0.35,
        wspace=0.45,
    )
    # Row 0 spans full width as two equal metadata columns.
    gs_meta = gs[0, :].subgridspec(1, 2, wspace=0.15)

    # Layout — row 0: metadata pair; row 1: two raw surfaces + results;
    # row 2: two filtered cell overlays + cell ACCF heatmap.
    ax_meta_reference = fig.add_subplot(gs_meta[0, 0])
    ax_meta_compared = fig.add_subplot(gs_meta[0, 1])
    ax_raw_ref = fig.add_subplot(gs[1, 0])
    ax_raw_comp = fig.add_subplot(gs[1, 1])
    ax_results = fig.add_subplot(gs[1, 2])
    ax_filtered_ref = fig.add_subplot(gs[2, 0])
    ax_filtered_comp = fig.add_subplot(gs[2, 1])
    ax_heatmap = fig.add_subplot(gs[2, 2])

    # Row 0: metadata tables
    draw_metadata_pair(
        ax_meta_reference,
        ax_meta_compared,
        metadata_reference,
        metadata_compared,
        noun="Surface",
        wrap_width=wrap_width,
    )

    # Row 1: raw surfaces + results metadata
    plot_depth_map_on_axes(
        ax_raw_ref,
        fig,
        mark_reference_raw.scan_image.data,
        mark_reference_raw.scan_image.scale_x,
        title="Reference Surface A",
    )
    plot_depth_map_on_axes(
        ax_raw_comp,
        fig,
        mark_compared_raw.scan_image.data,
        mark_compared_raw.scan_image.scale_x,
        title="Compared Surface B",
    )
    draw_metadata_box(
        ax_results,
        results_items,
        draw_border=False,
        wrap_width=wrap_width,
    )

    # Row 2: filtered surfaces with cell grid overlay + cell ACCF heatmap
    im_ref = plot_cell_overlay_on_axes(
        ax_filtered_ref,
        mark_reference_filtered.scan_image.data,
        mark_reference_filtered.scan_image.scale_x,
        cells=cells,
        cell_label_prefix="A",
        show_all_cells=True,
        space="reference",
    )
    _plot_surface_with_colorbar(
        fig, ax_filtered_ref, im_ref, title="Filtered Reference Surface A"
    )

    im_comp = plot_cell_overlay_on_axes(
        ax_filtered_comp,
        mark_compared_filtered.scan_image.data,
        mark_compared_filtered.scan_image.scale_x,
        cells=cells,
        cell_label_prefix="B",
        show_all_cells=False,
    )
    _plot_surface_with_colorbar(
        fig, ax_filtered_comp, im_comp, title="Filtered Compared Surface B"
    )

    ref_data = mark_reference_filtered.scan_image.data
    ref_sc = mark_reference_filtered.scan_image.scale_x
    heatmap_extent_um = (
        ref_data.shape[1] * ref_sc * mega,
        ref_data.shape[0] * ref_sc * mega,
    )
    _plot_cell_heatmap_on_axes(
        ax_heatmap,
        fig,
        cells=cells,
        cell_correlations=cell_correlations,
        surface_extent_um=heatmap_extent_um,
        cell_label_prefix="A",
    )

    return finish_overview(
        fig,
        tight_layout_kwargs={"pad": 0.8, "h_pad": 1.2, "w_pad": 0.8},
        subplots_adjust_kwargs={
            "left": 0.06,
            "right": 0.93,
            "top": 0.96,
            "bottom": 0.06,
        },
    )


def build_impression_results_metadata(
    cells: Sequence[Cell],
    cmc_result: ComparisonResult,
    comparison_params: ComparisonParams,
    mark_reference_raw: Mark,
    scale_x_um: float,
    scale_y_um: float,
) -> dict[str | Any, str | Any]:
    """Set up the overview of metadata to show in the plot."""
    results_items = {
        "Date report": datetime.now().strftime("%Y-%m-%d"),
        "Mark type": mark_reference_raw.mark_type.value,
        "Number of Cells": str(cmc_result.cell_count),
        "Number of CMCs": str(cmc_result.cmc_count),
        "CMC fraction": f"{cmc_result.cmc_fraction * 100:.2f} %",
        "CMC area fraction": f"{cmc_result.cmc_area_fraction * 100:.2f} %",
        "Data spacing (X)": f"{scale_x_um:.4f} µm",
        "Data spacing (Y)": f"{scale_y_um:.4f} µm",
        "Cell Size": f"{cells[0].cell_size_um[0]:.0f} µm",
        "Minimum cell similarity": f"{comparison_params.correlation_threshold}",
        "Max error cell position": f"{comparison_params.position_threshold * 1e6:.0f} µm",
        "Max error cell angle": f"{comparison_params.angle_deviation_threshold:.0f} degree",
    }
    return results_items
