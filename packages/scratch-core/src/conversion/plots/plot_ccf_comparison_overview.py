"""
Module for creating the CCF comparison overview plot.

Combines striation comparison plots (depth maps, side-by-side, metadata)
with score distribution analysis (histograms and LogLR transformation)
into a single overview figure.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from container_models.base import ImageRGB
from conversion.data_formats import Mark
from conversion.plots.data_formats import HistogramData, LlrTransformationData
from conversion.plots.plot_score_histograms import plot_score_histograms
from conversion.plots.plot_score_llr_transformation import plot_score_llr_transformation
from conversion.plots.utils import (
    draw_metadata_box,
    figure_to_array,
    get_height_ratios,
    get_metadata_dimensions,
    plot_depth_map_on_axes,
    plot_side_by_side_on_axes,
)


def _clear_colorbar_label(plot_ax: Axes) -> None:
    """Remove the colorbar label from the colorbar axes adjacent to plot_ax."""
    fig = plot_ax.get_figure()
    if fig is None:
        return
    n_before = len(fig.axes)
    # The colorbar axes is the one added right after plot_ax by make_axes_locatable.
    # It's the axes whose ylabel contains "Scan Depth" and sits to the right of plot_ax.
    plot_idx = fig.axes.index(plot_ax)
    for child_ax in fig.axes[plot_idx + 1 : min(plot_idx + 3, n_before)]:
        if child_ax.yaxis.label.get_text():
            child_ax.set_ylabel("")
            return


def plot_ccf_comparison_overview(
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    mark_reference_aligned: Mark,
    mark_compared_aligned: Mark,
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
    results_metadata: dict[str, str],
    histogram_data: HistogramData,
    histogram_data_transformed: HistogramData,
    llr_data: LlrTransformationData,
    wrap_width: int = 25,
) -> ImageRGB:
    """
    Create the CCF comparison overview figure.

    The figure layout (3 rows, 12-column grid):
    - Row 0: Metadata tables (Reference | Compared)
    - Row 1: Filtered ref | Filtered comp | Side-by-side | Results
    - Row 2: Score histograms | Transformed histograms | LogLR plot

    :param mark_reference_filtered: Filtered reference mark.
    :param mark_compared_filtered: Filtered compared mark.
    :param mark_reference_aligned: Aligned reference mark for side-by-side.
    :param mark_compared_aligned: Aligned compared mark for side-by-side.
    :param metadata_reference: Metadata dict for reference profile display.
    :param metadata_compared: Metadata dict for compared profile display.
    :param results_metadata: Results metadata dict for display.
    :param histogram_data: Input data for score histogram plot.
    :param histogram_data_transformed: Input data for transformed score histogram plot.
    :param llr_data: Input data for LogLR transformation plot.
    :param wrap_width: Maximum characters per line before wrapping metadata values.
    :returns: RGB image as uint8 array.
    """
    max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
        metadata_compared, metadata_reference, wrap_width
    )
    height_ratios = get_height_ratios(metadata_height_ratio, 0.38, 0.42)

    fig_height = 12 + (max_metadata_rows * 0.12)
    fig_height = max(10.0, min(15.0, fig_height))

    fig = plt.figure(figsize=(24, fig_height))
    gs = fig.add_gridspec(3, 12, height_ratios=height_ratios, hspace=0.35, wspace=0.7)

    ax_meta_ref = fig.add_subplot(gs[0, 0:6])
    draw_metadata_box(
        ax_meta_ref, metadata_reference, "Reference Profile (A)", wrap_width=wrap_width
    )

    ax_meta_comp = fig.add_subplot(gs[0, 6:])
    draw_metadata_box(
        ax_meta_comp, metadata_compared, "Compared Profile (B)", wrap_width=wrap_width
    )

    scale = mark_reference_filtered.scan_image.scale_x

    ax_heatmap_ref = fig.add_subplot(gs[1, 0:3])
    plot_depth_map_on_axes(
        ax_heatmap_ref,
        fig,
        mark_reference_filtered.scan_image.data,
        scale,
        "Filtered Reference Surface A",
        colorbar_width="3%",
        colorbar_pad=0.08,
        aspect="auto",
    )
    _clear_colorbar_label(ax_heatmap_ref)

    ax_heatmap_comp = fig.add_subplot(gs[1, 3:6])
    plot_depth_map_on_axes(
        ax_heatmap_comp,
        fig,
        mark_compared_filtered.scan_image.data,
        scale,
        "Filtered Compared Surface B",
        colorbar_width="3%",
        colorbar_pad=0.08,
        aspect="auto",
    )
    ax_heatmap_comp.set_ylabel("")
    _clear_colorbar_label(ax_heatmap_comp)

    ax_side = fig.add_subplot(gs[1, 6:9])
    plot_side_by_side_on_axes(
        ax_side,
        fig,
        mark_reference_aligned.scan_image.data,
        mark_compared_aligned.scan_image.data,
        scale,
        title="Surface A / Moved Surface B",
        colorbar_width="3%",
        colorbar_pad=0.08,
        aspect="auto",
    )
    ax_side.set_ylabel("")

    ax_results = fig.add_subplot(gs[1, 10:])
    draw_metadata_box(
        ax_results, results_metadata, draw_border=False, wrap_width=wrap_width
    )

    ax_hist = fig.add_subplot(gs[2, 0:4])
    plot_score_histograms(ax_hist, histogram_data)
    ax_hist.set_title("Score histograms", fontsize=12, fontweight="bold")

    ax_hist_trans = fig.add_subplot(gs[2, 4:8])
    plot_score_histograms(ax_hist_trans, histogram_data_transformed)
    ax_hist_trans.set_title(
        "Transformed score histograms", fontsize=12, fontweight="bold"
    )

    ax_llr = fig.add_subplot(gs[2, 8:])
    plot_score_llr_transformation(ax_llr, llr_data)

    fig.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.5)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.06)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr
