from typing import Sequence

import matplotlib.pyplot as plt

from container_models.base import ImageRGB
from conversion.data_formats import Mark, MarkMetadata
from plots.likelihood_ratio.data_formats import (
    HistogramData,
    LlrTransformationData,
)
from plots.cell_overlays import plot_cell_overlay_on_axes
from plots.likelihood_ratio.distributions import (
    plot_score_histograms,
    plot_score_llr_transformation,
)
from plots.utils import (
    finish_overview,
    get_height_ratios,
    overview_figure_height,
)
from plots.on_axes import _plot_surface_with_colorbar
from plots.metadata_tables import (
    get_metadata_dimensions,
    draw_metadata_box,
    draw_metadata_pair,
)
from conversion.surface_comparison.models import Cell


def plot_cmc_comparison_overview(
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    cells: Sequence[Cell],
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
    results_metadata: dict[str, str],
    histogram_data: HistogramData,
    llr_data: LlrTransformationData,
    wrap_width: int = 25,
) -> ImageRGB:
    """
    Generate the CMC + LR overview figure.

    Combines metadata tables, filtered surface visualizations with cell
    overlays, score histograms, and a LogLR transformation plot into a
    single overview figure.

    :param mark_reference_filtered: Reference mark after filtering.
    :param mark_compared_filtered: Compared mark after filtering.
    :param cells: Cells to plot.
    :param metadata_reference: Metadata object for reference mark display.
    :param metadata_compared: Metadata object for compared mark display.
    :param results_metadata: Results metadata dict for display.
    :param histogram_data: Input data for score histogram plot.
    :param llr_data: Input data for LogLR transformation plot.
    :param wrap_width: Maximum characters per line before wrapping metadata table values.
    :returns: RGB image as uint8 array.
    """
    max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
        metadata_compared, metadata_reference, wrap_width
    )
    height_ratios = get_height_ratios(metadata_height_ratio, 0.40, 0.40)

    fig_height = overview_figure_height(max_metadata_rows, 12, 10.0, 15.0)

    fig = plt.figure(figsize=(16, fig_height))

    gs = fig.add_gridspec(3, 6, height_ratios=height_ratios, hspace=0.35, wspace=0.45)

    # Layout — row 0: metadata pair; row 1: two cell surfaces + results; row 2: score histograms + LogLR plot.
    ax_meta_ref = fig.add_subplot(gs[0, 0:3])
    ax_meta_comp = fig.add_subplot(gs[0, 3:6])
    ax_surf_ref = fig.add_subplot(gs[1, 0:2])
    ax_surf_comp = fig.add_subplot(gs[1, 2:4])
    ax_results = fig.add_subplot(gs[1, 4:6])
    ax_hist = fig.add_subplot(gs[2, 0:3])
    ax_llr = fig.add_subplot(gs[2, 3:6])

    # Row 0: metadata tables
    draw_metadata_pair(
        ax_meta_ref,
        ax_meta_comp,
        metadata_reference,
        metadata_compared,
        noun="Surface",
        wrap_width=wrap_width,
    )

    # Row 1: filtered surfaces with cell overlay + results metadata
    im_ref = plot_cell_overlay_on_axes(
        ax_surf_ref,
        mark_reference_filtered.scan_image.data,
        mark_reference_filtered.scan_image.scale_x,
        cells=cells,
        cell_label_prefix="A",
        show_all_cells=True,
        space="reference",
    )
    _plot_surface_with_colorbar(
        fig, ax_surf_ref, im_ref, "Filtered Reference Surface A"
    )

    im_comp = plot_cell_overlay_on_axes(
        ax_surf_comp,
        mark_compared_filtered.scan_image.data,
        mark_compared_filtered.scan_image.scale_x,
        cells=cells,
        cell_label_prefix="B",
        show_all_cells=False,
    )
    _plot_surface_with_colorbar(
        fig, ax_surf_comp, im_comp, "Filtered, Moved Compared Surface B"
    )

    draw_metadata_box(
        ax_results, results_metadata, draw_border=False, wrap_width=wrap_width
    )

    # Row 2: score histograms + LogLR plot
    plot_score_histograms(ax_hist, histogram_data)
    plot_score_llr_transformation(ax_llr, llr_data)

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
