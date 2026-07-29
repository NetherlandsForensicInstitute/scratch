"""
Module for creating the CCF comparison overview plot.

Combines striation comparison plots (depth maps, side-by-side, metadata)
with score distribution analysis (histograms and LogLR transformation)
into a single overview figure.
"""

import matplotlib.pyplot as plt

from container_models.base import ImageRGB
from conversion.data_formats import Mark, MarkMetadata
from conversion.plots.likelihood_ratio.data_formats import (
    HistogramData,
    LlrTransformationData,
)
from conversion.plots.likelihood_ratio.distributions import (
    plot_score_histograms,
    plot_score_llr_transformation,
)
from conversion.plots.utils import (
    finish_overview,
    get_height_ratios,
    overview_figure_height,
)
from conversion.plots.on_axes import plot_side_by_side_on_axes, plot_depth_map_on_axes
from conversion.plots.metadata_tables import (
    get_metadata_dimensions,
    draw_metadata_box,
    draw_metadata_pair,
)


def plot_ccf_comparison_overview(
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    mark_reference_aligned: Mark,
    mark_compared_aligned: Mark,
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
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
    :param metadata_reference: Metadata object for reference profile display.
    :param metadata_compared: Metadata object for compared profile display.
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

    fig_height = overview_figure_height(max_metadata_rows, 12, 10.0, 15.0)

    fig = plt.figure(figsize=(24, fig_height))
    gs = fig.add_gridspec(3, 12, height_ratios=height_ratios, hspace=0.35, wspace=0.7)

    # Layout — row 0: metadata pair; row 1: filtered ref | filtered comp |
    # side-by-side | results; row 2: histograms | transformed | LogLR.
    ax_meta_ref = fig.add_subplot(gs[0, 0:6])
    ax_meta_comp = fig.add_subplot(gs[0, 6:])
    ax_heatmap_ref = fig.add_subplot(gs[1, 0:3])
    ax_heatmap_comp = fig.add_subplot(gs[1, 3:6])
    ax_side = fig.add_subplot(gs[1, 6:9])
    ax_results = fig.add_subplot(gs[1, 10:])
    ax_hist = fig.add_subplot(gs[2, 0:4])
    ax_hist_trans = fig.add_subplot(gs[2, 4:8])
    ax_llr = fig.add_subplot(gs[2, 8:])

    scale = mark_reference_filtered.scan_image.scale_x

    # Row 0: metadata tables
    draw_metadata_pair(
        ax_meta_ref,
        ax_meta_comp,
        metadata_reference,
        metadata_compared,
        noun="Profile",
        wrap_width=wrap_width,
    )

    # Row 1: filtered surfaces + side-by-side + results metadata
    plot_depth_map_on_axes(
        ax_heatmap_ref,
        fig,
        mark_reference_filtered.scan_image.data,
        scale,
        "Filtered Reference Surface A",
        colorbar_label=None,
        colorbar_width="3%",
        colorbar_pad=0.08,
        aspect="auto",
    )

    plot_depth_map_on_axes(
        ax_heatmap_comp,
        fig,
        mark_compared_filtered.scan_image.data,
        scale,
        "Filtered Compared Surface B",
        colorbar_label=None,
        colorbar_width="3%",
        colorbar_pad=0.08,
        aspect="auto",
    )
    ax_heatmap_comp.set_ylabel("")

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

    draw_metadata_box(
        ax_results, results_metadata, draw_border=False, wrap_width=wrap_width
    )

    # Row 2: score histograms + transformed histograms + LogLR plot
    plot_score_histograms(ax_hist, histogram_data)
    plot_score_histograms(
        ax_hist_trans, histogram_data_transformed, title="Transformed score histograms"
    )
    plot_score_llr_transformation(ax_llr, llr_data)

    return finish_overview(
        fig,
        tight_layout_kwargs={"pad": 1.0, "h_pad": 1.5, "w_pad": 1.5},
        subplots_adjust_kwargs={
            "left": 0.04,
            "right": 0.96,
            "top": 0.96,
            "bottom": 0.06,
        },
    )
