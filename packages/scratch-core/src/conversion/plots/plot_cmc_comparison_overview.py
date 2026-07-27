from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from container_models.base import ImageRGB
from conversion.data_formats import Mark, MarkMetadata
from conversion.plots.data_formats import (
    HistogramData,
    LlrTransformationData,
)
from conversion.plots.plot_impression import (
    plot_cell_overlay_on_axes, _robust_color_limits,
)
from conversion.plots.plot_score_histograms import plot_score_histograms
from conversion.plots.plot_score_llr_transformation import plot_score_llr_transformation
from conversion.plots.utils import (
    draw_metadata_box,
    figure_to_array,
    get_height_ratios,
    get_metadata_dimensions,
)
from conversion.surface_comparison.models import Cell


def _plot_surface_with_colorbar(
    fig: Figure,
    ax: Axes,
    im: AxesImage,
    title: str,
    color_sigma: float = 3.0,
    extendfrac: float = 0.08,
) -> None:
    """
    Plot a cell overlay on axes and add an outlier-aware colorbar.

    The image is (re)clipped to mean ± color_sigma*std so the surface
    colours and the colorbar agree: the main body spans the clipped range,
    red lines mark that clipping boundary, and the true (unclipped) data
    min/max appear at the tips of the extend triangles. Everything is read
    back off ``im`` itself, so no data array needs to be passed in.

    :param fig: Figure to attach the colorbar to.
    :param ax: Axes the image was plotted on.
    :param im: AxesImage returned by plot_cell_overlay_on_axes.
    :param title: Axes title.
    :param color_sigma: Std multiplier for the clip bounds (red lines).
    :param extendfrac: Fraction of the colorbar length per extend triangle.
    """
    ax.set_title(title, fontsize=12, fontweight="bold")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    arr = np.ma.masked_invalid(im.get_array())

    # Clip bounds (mean ± k*std), applied to the image so the *surface*
    # colours are clipped too — this is what puts the red line at ~1.27
    # instead of at the true max. If the image is already clipped upstream
    # this is effectively a no-op, so it's safe to keep either way.
    vmin, vmax = _robust_color_limits(arr.filled(np.nan), k=color_sigma)
    im.set_clim(vmin, vmax)

    cbar = fig.colorbar(
        im, cax=cax, label="Scan Depth [µm]", extend="both", extendfrac=extendfrac
    )
    cbar.ax.tick_params(labelsize=9)

    # Red lines at the actual clip bounds (now ~±1.27, not the true max).
    cbar.ax.axhline(vmin, color="red", linewidth=2)
    cbar.ax.axhline(vmax, color="red", linewidth=2)

    # Keep in-range ticks (dropping any that would crowd the tips), then add
    # the true min/max at the triangle tips.
    margin = 0.06 * (vmax - vmin)
    default_ticks = [
        t for t in cbar.get_ticks() if vmin + margin <= t <= vmax - margin
    ]
    cbar.set_ticks([vmin, *default_ticks, vmax])
    cbar.set_ticklabels(
        [f"{vmin:.2f}", *[f"{t:.2f}" for t in default_ticks], f"{vmax:.2f}"]
    )


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

    fig_height = 12 + (max_metadata_rows * 0.12)
    fig_height = max(10.0, min(15.0, fig_height))

    fig = plt.figure(figsize=(16, fig_height))

    gs = fig.add_gridspec(3, 6, height_ratios=height_ratios, hspace=0.35, wspace=0.45)

    # Row 0: Metadata tables (2 equal halves)
    ax_meta_ref = fig.add_subplot(gs[0, 0:3])
    draw_metadata_box(
        ax_meta_ref,
        metadata_reference.to_display_dict(),
        "Reference Surface (A)",
        wrap_width=wrap_width,
    )

    ax_meta_comp = fig.add_subplot(gs[0, 3:6])
    draw_metadata_box(
        ax_meta_comp,
        metadata_compared.to_display_dict(),
        "Compared Surface (B)",
        wrap_width=wrap_width,
    )

    # Row 1: Filtered surfaces with cell overlay + results metadata (3 equal thirds)
    ax_filtered_ref = fig.add_subplot(gs[1, 0:2])
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
        fig, ax_filtered_ref, im_ref, "Filtered Reference Surface A"
    )

    ax_filtered_comp = fig.add_subplot(gs[1, 2:4])

    im_comp = plot_cell_overlay_on_axes(
        ax_filtered_comp,
        mark_compared_filtered.scan_image.data,
        mark_compared_filtered.scan_image.scale_x,
        cells=cells,
        cell_label_prefix="B",
        show_all_cells=False,
    )
    _plot_surface_with_colorbar(
        fig, ax_filtered_comp, im_comp, "Filtered, Moved Compared Surface B"
    )

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
