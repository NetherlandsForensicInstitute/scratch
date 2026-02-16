"""
Module for creating the CCF comparison overview plot.

Combines striation comparison plots (depth maps, side-by-side, metadata)
with score distribution analysis (histograms and LogLR transformation)
into a single overview figure.
"""

import matplotlib.pyplot as plt

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

    The figure layout (3 rows, 6-column grid):
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

    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(3, 6, height_ratios=height_ratios, hspace=0.35, wspace=0.45)

    ax_meta_ref = fig.add_subplot(gs[0, 0:3])
    draw_metadata_box(
        ax_meta_ref, metadata_reference, "Reference Profile (A)", wrap_width=wrap_width
    )

    ax_meta_comp = fig.add_subplot(gs[0, 3:6])
    draw_metadata_box(
        ax_meta_comp, metadata_compared, "Compared Profile (B)", wrap_width=wrap_width
    )

    scale = mark_reference_filtered.scan_image.scale_x

    ax_heatmap_ref = fig.add_subplot(gs[1, 0:1])
    plot_depth_map_on_axes(
        ax_heatmap_ref,
        fig,
        mark_reference_filtered.scan_image.data,
        scale,
        "Filtered Reference Surface A",
    )

    ax_heatmap_comp = fig.add_subplot(gs[1, 1:2])
    plot_depth_map_on_axes(
        ax_heatmap_comp,
        fig,
        mark_compared_filtered.scan_image.data,
        scale,
        "Filtered Compared Surface B",
    )

    ax_side = fig.add_subplot(gs[1, 2:4])
    plot_side_by_side_on_axes(
        ax_side,
        fig,
        mark_reference_aligned.scan_image.data,
        mark_compared_aligned.scan_image.data,
        scale,
    )

    ax_results = fig.add_subplot(gs[1, 4:6])
    draw_metadata_box(
        ax_results, results_metadata, draw_border=False, wrap_width=wrap_width
    )

    ax_hist = fig.add_subplot(gs[2, 0:2])
    plot_score_histograms(ax_hist, histogram_data)
    ax_hist.set_title("Score histograms", fontsize=12, fontweight="bold")

    ax_hist_trans = fig.add_subplot(gs[2, 2:4])
    plot_score_histograms(ax_hist_trans, histogram_data_transformed)
    ax_hist_trans.set_title(
        "Transformed score histograms", fontsize=12, fontweight="bold"
    )

    ax_llr = fig.add_subplot(gs[2, 4:6])
    plot_score_llr_transformation(ax_llr, llr_data)

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.8)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.96, bottom=0.06)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr
