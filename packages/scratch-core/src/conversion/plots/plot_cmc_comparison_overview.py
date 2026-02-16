import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from container_models.base import ImageRGB
from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    HistogramData,
    ImpressionComparisonMetrics,
    LlrTransformationData,
)
from conversion.plots.plot_impression import (
    compute_cell_size_um,
    plot_cell_overlay_on_axes,
)
from conversion.plots.plot_score_histograms import plot_score_histograms
from conversion.plots.plot_score_llr_transformation import plot_score_llr_transformation
from conversion.plots.utils import (
    draw_metadata_box,
    figure_to_array,
    get_height_ratios,
    get_metadata_dimensions,
)


def plot_cmc_comparison_overview(
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    metrics: ImpressionComparisonMetrics,
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
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
    :param metrics: Comparison metrics including correlation values.
    :param metadata_reference: Metadata dict for reference mark display.
    :param metadata_compared: Metadata dict for compared mark display.
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

    fig_height = 12 + (max_metadata_rows * 0.12)
    fig_height = max(10.0, min(15.0, fig_height))

    fig = plt.figure(figsize=(16, fig_height))

    gs = fig.add_gridspec(3, 6, height_ratios=height_ratios, hspace=0.35, wspace=0.45)

    # Row 0: Metadata tables (2 equal halves)
    ax_meta_ref = fig.add_subplot(gs[0, 0:3])
    draw_metadata_box(
        ax_meta_ref, metadata_reference, "Reference Surface (A)", wrap_width=wrap_width
    )

    ax_meta_comp = fig.add_subplot(gs[0, 3:6])
    draw_metadata_box(
        ax_meta_comp, metadata_compared, "Compared Surface (B)", wrap_width=wrap_width
    )

    # Row 1: Filtered surfaces with cell overlay + results metadata (3 equal thirds)
    ax_filtered_ref = fig.add_subplot(gs[1, 0:2])
    im_ref = plot_cell_overlay_on_axes(
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

    ax_filtered_comp = fig.add_subplot(gs[1, 2:4])
    cell_size_um = compute_cell_size_um(
        mark_reference_filtered.scan_image.data.shape,
        mark_reference_filtered.scan_image.scale_x,
        metrics.cell_correlations.shape,
    )
    im_comp = plot_cell_overlay_on_axes(
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
        "Filtered, Moved Compared Surface B", fontsize=12, fontweight="bold"
    )
    divider_comp = make_axes_locatable(ax_filtered_comp)
    cax_comp = divider_comp.append_axes("right", size="5%", pad=0.05)
    cbar_comp = fig.colorbar(im_comp, cax=cax_comp, label="Scan Depth [µm]")
    cbar_comp.ax.tick_params(labelsize=9)

    ax_results = fig.add_subplot(gs[1, 4:6])
    draw_metadata_box(
        ax_results, results_metadata, draw_border=False, wrap_width=wrap_width
    )

    # Row 2: Score histograms + LogLR plot (2 equal halves)
    ax_hist = fig.add_subplot(gs[2, 0:3])
    plot_score_histograms(ax_hist, histogram_data)
    ax_hist.set_title("Score histograms", fontsize=12, fontweight="bold")

    ax_llr = fig.add_subplot(gs[2, 3:6])
    plot_score_llr_transformation(ax_llr, llr_data)

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.8)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.96, bottom=0.06)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr
