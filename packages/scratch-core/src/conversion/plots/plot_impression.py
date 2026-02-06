"""
Impression mark comparison visualization.

Translates MATLAB functions:
- GenerateAdditionalNISTFigures.m (orchestrator)
- PlotResultsAreaNIST.m (area-based correlation plots)
- PlotResultsCmcNIST.m (cell/CMC-based correlation plots)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d

from container_models.base import FloatArray2D, ImageRGB
from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    ImpressionComparisonMetrics,
    ImpressionComparisonPlots,
)
from conversion.plots.utils import (
    DEFAULT_COLORMAP,
    figure_to_array,
    get_figure_dimensions,
    plot_depth_map_on_axes,
)


def plot_impression_comparison_results(
    mark_reference_leveled: Mark,
    mark_compared_leveled: Mark,
    mark_reference_filtered: Mark,
    mark_compared_filtered: Mark,
    metrics: ImpressionComparisonMetrics,
    _metadata_reference: dict[str, str],
    _metadata_compared: dict[str, str],
) -> ImpressionComparisonPlots:
    """
    Generate visualization results for impression mark comparison.

    Main orchestrator function equivalent to MATLAB GenerateAdditionalNISTFigures.m.
    Generates both area-based and cell/CMC-based visualizations based on which
    results are available in the metrics.

    :param mark_reference_leveled: Reference mark after leveling.
    :param mark_compared_leveled: Compared mark after leveling.
    :param mark_reference_filtered: Reference mark after filtering.
    :param mark_compared_filtered: Compared mark after filtering.
    :param metrics: Comparison metrics including correlation values.
    :param _metadata_reference: Metadata dict for reference mark display (reserved for future use).
    :param _metadata_compared: Metadata dict for compared mark display (reserved for future use).
    :returns: ImpressionComparisonPlots with all rendered images.
    """
    # Initialize all plots as None
    leveled_ref = None
    leveled_comp = None
    filtered_ref = None
    filtered_comp = None
    difference_map = None
    area_xcorr = None
    cell_ref = None
    cell_comp = None
    cell_overlay = None
    cell_xcorr = None
    cell_histogram = None

    # Generate area-based plots if available
    if metrics.has_area_results:
        (
            leveled_ref,
            leveled_comp,
            filtered_ref,
            filtered_comp,
            difference_map,
            area_xcorr,
        ) = plot_area_figures(
            mark_ref_leveled=mark_reference_leveled,
            mark_comp_leveled=mark_compared_leveled,
            mark_ref_filtered=mark_reference_filtered,
            mark_comp_filtered=mark_compared_filtered,
            correlation_value=metrics.area_correlation,
        )

    # Generate cell/CMC-based plots if available
    if metrics.has_cell_results:
        (
            cell_ref,
            cell_comp,
            cell_overlay,
            cell_xcorr,
            cell_histogram,
        ) = plot_cmc_figures(
            mark_ref_filtered=mark_reference_filtered,
            mark_comp_filtered=mark_compared_filtered,
            cell_correlations=metrics.cell_correlations,
        )

    return ImpressionComparisonPlots(
        leveled_reference=leveled_ref,
        leveled_compared=leveled_comp,
        filtered_reference=filtered_ref,
        filtered_compared=filtered_comp,
        difference_map=difference_map,
        area_cross_correlation=area_xcorr,
        cell_reference=cell_ref,
        cell_compared=cell_comp,
        cell_overlay=cell_overlay,
        cell_cross_correlation=cell_xcorr,
        cell_correlation_histogram=cell_histogram,
    )


def plot_area_figures(
    mark_ref_leveled: Mark,
    mark_comp_leveled: Mark,
    mark_ref_filtered: Mark,
    mark_comp_filtered: Mark,
    correlation_value: float,
) -> tuple[ImageRGB, ImageRGB, ImageRGB, ImageRGB, ImageRGB, ImageRGB]:
    """
    Generate 6 area-based plots for impression comparison.

    Equivalent to MATLAB PlotResultsAreaNIST.m.
    Generates:
    1. Leveled reference surface
    2. Leveled compared surface
    3. Filtered reference surface
    4. Filtered compared surface
    5. Difference map (compared - reference)
    6. Cross-correlation surface

    :param mark_ref_leveled: Reference mark after leveling.
    :param mark_comp_leveled: Compared mark after leveling.
    :param mark_ref_filtered: Reference mark after filtering.
    :param mark_comp_filtered: Compared mark after filtering.
    :param correlation_value: Areal correlation coefficient.
    :returns: Tuple of 6 ImageRGB arrays.
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

    # 5. Difference map
    diff_map = plot_difference_map(
        data_ref=mark_ref_filtered.scan_image.data,
        data_comp=mark_comp_filtered.scan_image.data,
        scale=scale_ref,
    )

    # 6. Cross-correlation surface
    xcorr = plot_cross_correlation_surface(
        data_ref=mark_ref_filtered.scan_image.data,
        data_comp=mark_comp_filtered.scan_image.data,
        scale=scale_ref,
        correlation_value=correlation_value,
    )

    return leveled_ref, leveled_comp, filtered_ref, filtered_comp, diff_map, xcorr


def plot_cmc_figures(
    mark_ref_filtered: Mark,
    mark_comp_filtered: Mark,
    cell_correlations: FloatArray2D,
) -> tuple[ImageRGB, ImageRGB, ImageRGB, ImageRGB, ImageRGB]:
    """
    Generate 5 CMC/cell-based plots for impression comparison.

    Equivalent to MATLAB PlotResultsCmcNIST.m.
    Generates:
    1. Cell-preprocessed reference
    2. Cell-preprocessed compared
    3. All cells overlay visualization
    4. Cell cross-correlation heatmap
    5. Cell correlation histogram

    :param mark_ref_filtered: Reference mark after filtering.
    :param mark_comp_filtered: Compared mark after filtering.
    :param cell_correlations: Grid of per-cell correlation values.
    :returns: Tuple of 5 ImageRGB arrays.
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

    # 5. Cell correlation histogram
    cell_histogram = plot_correlation_histogram(
        cell_correlations=cell_correlations,
    )

    return cell_ref, cell_comp, cell_overlay, cell_xcorr, cell_histogram


def plot_depth_map_with_axes(
    data: FloatArray2D,
    scale: float,
    title: str,
) -> ImageRGB:
    """
    Plot a depth map with axes and colorbar.

    :param data: Depth data in meters.
    :param scale: Pixel scale in meters.
    :param title: Title for the plot.
    :returns: RGB image as uint8 array.
    """
    height, width = data.shape
    fig_height, fig_width = get_figure_dimensions(height, width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_depth_map_on_axes(ax, fig, data, scale, title)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_difference_map(
    data_ref: FloatArray2D,
    data_comp: FloatArray2D,
    scale: float,
) -> ImageRGB:
    """
    Plot the difference map between two surfaces.

    :param data_ref: Reference surface data in meters.
    :param data_comp: Compared surface data in meters.
    :param scale: Pixel scale in meters.
    :returns: RGB image as uint8 array.
    """
    # Compute difference (handle NaN values)
    diff = data_comp - data_ref

    # Compute Sq of difference (RMS of valid values)
    valid_diff = diff[~np.isnan(diff)]
    sq_diff = np.sqrt(np.mean(valid_diff**2)) * 1e6 if len(valid_diff) > 0 else 0.0

    height, width = diff.shape
    fig_height, fig_width = get_figure_dimensions(height, width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    extent = (0, width * scale * 1e6, 0, height * scale * 1e6)
    im = ax.imshow(
        diff * 1e6,
        cmap="RdBu_r",  # Diverging colormap centered at 0
        aspect="equal",
        origin="lower",
        extent=extent,
    )

    # Center colormap at 0
    vmax = np.nanmax(np.abs(diff * 1e6))
    im.set_clim(-vmax, vmax)

    ax.set_xlabel("X - Position [um]", fontsize=11)
    ax.set_ylabel("Y - Position [um]", fontsize=11)
    ax.set_title(
        f"Difference Map (Sq = {sq_diff:.4f} um)", fontsize=12, fontweight="bold"
    )
    ax.tick_params(labelsize=10)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, label="Difference [um]")
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_cross_correlation_surface(
    data_ref: FloatArray2D,
    data_comp: FloatArray2D,
    scale: float,
    correlation_value: float,
) -> ImageRGB:
    """
    Plot the 2D cross-correlation surface.

    :param data_ref: Reference surface data in meters.
    :param data_comp: Compared surface data in meters.
    :param scale: Pixel scale in meters.
    :param correlation_value: Pre-computed correlation coefficient.
    :returns: RGB image as uint8 array.
    """
    # Replace NaN with 0 for correlation computation
    ref_clean = np.nan_to_num(data_ref, nan=0.0)
    comp_clean = np.nan_to_num(data_comp, nan=0.0)

    # Normalize for correlation
    ref_norm = ref_clean - np.mean(ref_clean)
    comp_norm = comp_clean - np.mean(comp_clean)

    # Compute 2D cross-correlation (use 'same' mode for same-size output)
    xcorr = correlate2d(ref_norm, comp_norm, mode="same", boundary="fill", fillvalue=0)

    # Normalize to correlation coefficient scale
    norm_factor = np.sqrt(np.sum(ref_norm**2) * np.sum(comp_norm**2))
    if norm_factor > 0:
        xcorr = xcorr / norm_factor

    height, width = xcorr.shape
    fig_height, fig_width = get_figure_dimensions(height, width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create extent in lag coordinates (centered at 0)
    half_h = height // 2
    half_w = width // 2
    extent_um = (
        -half_w * scale * 1e6,
        half_w * scale * 1e6,
        -half_h * scale * 1e6,
        half_h * scale * 1e6,
    )

    im = ax.imshow(
        xcorr,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin="lower",
        extent=extent_um,
    )

    ax.set_xlabel("X - Lag [um]", fontsize=11)
    ax.set_ylabel("Y - Lag [um]", fontsize=11)
    ax.set_title(
        f"Cross-Correlation (Max = {correlation_value:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(labelsize=10)

    # Mark the peak
    peak_idx = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    peak_y = (peak_idx[0] - half_h) * scale * 1e6
    peak_x = (peak_idx[1] - half_w) * scale * 1e6
    ax.plot(peak_x, peak_y, "r+", markersize=15, markeredgewidth=2)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, label="Correlation")
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
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
    n_rows, n_cols = cell_correlations.shape

    fig_height, fig_width = get_figure_dimensions(height, width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

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

    ax.set_xlabel("X - Position [um]", fontsize=11)
    ax.set_ylabel("Y - Position [um]", fontsize=11)
    ax.set_title("Cell Grid with Correlation Values", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)

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

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label="Correlation")
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_correlation_histogram(
    cell_correlations: FloatArray2D,
    threshold: float = 0.5,
) -> ImageRGB:
    """
    Plot histogram of per-cell correlation values.

    :param cell_correlations: Grid of per-cell correlation values.
    :param threshold: CMC threshold to mark on histogram.
    :returns: RGB image as uint8 array.
    """
    # Flatten and remove NaN values
    valid_correlations = cell_correlations.flatten()
    valid_correlations = valid_correlations[~np.isnan(valid_correlations)]

    # Count cells above threshold
    n_above = np.sum(valid_correlations >= threshold)
    n_total = len(valid_correlations)
    cmc_score = (n_above / n_total * 100) if n_total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create histogram
    n_bins = 20
    _, _, patches = ax.hist(
        valid_correlations,
        bins=n_bins,
        range=(0, 1),
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
    )

    # Color bars above threshold differently (patches is BarContainer for single input)
    for patch in patches:  # type: ignore[union-attr]
        bin_center = patch.get_x() + patch.get_width() / 2
        if bin_center >= threshold:
            patch.set_facecolor("forestgreen")

    # Add threshold line
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"CMC Threshold = {threshold:.2f}",
    )

    ax.set_xlabel("Correlation Coefficient", fontsize=11)
    ax.set_ylabel("Number of Cells", fontsize=11)
    ax.set_title(
        f"Cell Correlation Distribution (CMC = {cmc_score:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics annotation
    stats_text = (
        f"N = {n_total}\n"
        f"Mean = {np.mean(valid_correlations):.3f}\n"
        f"Std = {np.std(valid_correlations):.3f}\n"
        f"Above threshold: {n_above}/{n_total}"
    )
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr
