import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from container_models.base import FloatArray1D, ImageRGB
from conversion.data_formats import Mark
from conversion.plots.data_formats import ImpressionComparisonMetrics
from conversion.plots.plot_impression import _plot_cell_overlay_on_axes
from conversion.plots.plot_score_histograms import DensityDict, plot_score_histograms
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
    scores: FloatArray1D,
    labels: FloatArray1D,
    bins: int | None,
    densities: DensityDict | None,
    new_score: float | None,
    llr_scores: FloatArray1D,
    llrs: FloatArray1D,
    llrs_at5: FloatArray1D,
    llrs_at95: FloatArray1D,
    score_llr_point: tuple[float, float] | None,
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
    :param results_metadata: Pre-built results metadata dict for display.
    :param scores: Score values for histogram.
    :param labels: Labels (0 for KNM, 1 for KM) for histogram.
    :param bins: Number of bins for histogram. If None, uses 'auto' binning.
    :param densities: Optional density curves for histogram overlay.
    :param new_score: Optional score value to mark on the histogram.
    :param llr_scores: Score axis for the LogLR plot.
    :param llrs: LogLR values.
    :param llrs_at5: LogLR values at 5% confidence.
    :param llrs_at95: LogLR values at 95% confidence.
    :param score_llr_point: Optional (score, llr) coordinate to mark on the LogLR plot.
    :param wrap_width: Maximum characters per line before wrapping.
    :returns: RGB image as uint8 array.
    """
    max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
        metadata_compared, metadata_reference, wrap_width
    )
    height_ratios = get_height_ratios(metadata_height_ratio, 0.40, 0.40)

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

    # Row 0: Metadata tables
    gs_meta = gs[0, :].subgridspec(1, 2, wspace=0.15)

    ax_meta_ref = fig.add_subplot(gs_meta[0, 0])
    draw_metadata_box(
        ax_meta_ref, metadata_reference, "Reference Surface (A)", wrap_width=wrap_width
    )

    ax_meta_comp = fig.add_subplot(gs_meta[0, 1])
    draw_metadata_box(
        ax_meta_comp, metadata_compared, "Compared Surface (B)", wrap_width=wrap_width
    )

    # Row 1: Filtered surfaces with cell overlay + results metadata
    ax_filtered_ref = fig.add_subplot(gs[1, 0])
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

    ax_filtered_comp = fig.add_subplot(gs[1, 1])
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
        "Filtered, Moved Compared Surface B", fontsize=12, fontweight="bold"
    )
    divider_comp = make_axes_locatable(ax_filtered_comp)
    cax_comp = divider_comp.append_axes("right", size="5%", pad=0.05)
    cbar_comp = fig.colorbar(im_comp, cax=cax_comp, label="Scan Depth [µm]")
    cbar_comp.ax.tick_params(labelsize=9)

    ax_results = fig.add_subplot(gs[1, 2])
    draw_metadata_box(
        ax_results, results_metadata, draw_border=False, wrap_width=wrap_width
    )

    # Row 2: Score histograms + LogLR plot
    gs_bottom = gs[2, :].subgridspec(1, 2, wspace=0.30)

    ax_hist = fig.add_subplot(gs_bottom[0, 0])
    plot_score_histograms(scores, labels, ax_hist, bins, densities, new_score)
    ax_hist.set_title("Score histograms", fontsize=12, fontweight="bold")

    ax_llr = fig.add_subplot(gs_bottom[0, 1])
    plot_score_llr_transformation(
        ax_llr, llr_scores, llrs, llrs_at5, llrs_at95, score_llr_point
    )

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.8)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.96, bottom=0.06)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr
