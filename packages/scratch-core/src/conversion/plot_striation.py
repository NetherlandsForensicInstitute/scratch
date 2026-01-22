from dataclasses import dataclass
from typing import Optional
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from conversion.data_formats import Mark

# Default plot settings
DEFAULT_COLORMAP = "viridis"
DEFAULT_FIGSIZE_PROFILES = (10, 4)
DEFAULT_FIGSIZE_PROFILES_XCORR = (10, 8)
DEFAULT_FIGSIZE_OVERVIEW = (16, 14)


@dataclass
class CorrelationMetrics:
    """Metrics from profile correlation comparison for display."""

    score: float
    """Correlation coefficient."""

    shift: float
    """Shift in µm."""

    overlap: float
    """Overlap percentage."""

    sq_a: Optional[float] = None
    """Sq (RMS roughness) of reference surface A in µm."""

    sq_b: Optional[float] = None
    """Sq (RMS roughness) of compared surface B in µm."""

    sq_b_minus_a: Optional[float] = None
    """Sq of difference (B-A) in µm."""

    sq_ratio: Optional[float] = None
    """Sq(B) / Sq(A) percentage."""

    sign_diff_dsab: Optional[float] = None
    """Signed difference DsAB percentage."""

    data_spacing: Optional[float] = None
    """Data spacing in µm."""

    cutoff_low_pass: Optional[float] = None
    """Cutoff length low-pass filter in µm."""

    cutoff_high_pass: Optional[float] = None
    """Cutoff length high-pass filter in µm."""

    date_report: Optional[str] = None
    """Date of report."""

    mark_type: Optional[str] = None
    """Type of mark."""


@dataclass
class WavelengthXcorrData:
    """Pre-computed wavelength-dependent cross-correlation data for plotting."""

    xcorr_full_range: float
    """Full range correlation coefficient."""

    quality_passbands: np.ndarray
    """(N, 2) array of [low, high] wavelength bands in µm."""

    range_dependent_xcorrs: np.ndarray
    """(N,) array of correlations per band."""


@dataclass
class WavelengthXcorrResult:
    """Results from wavelength-dependent cross-correlation plotting."""

    image: np.ndarray
    """Rendered figure as (H, W, 3) uint8."""

    data: WavelengthXcorrData
    """The input data that was plotted."""


@dataclass
class StriationComparisonResults:
    """
    Results from striation (profile) comparison visualization.

    All image fields are numpy arrays with shape (H, W, 3) uint8.
    """

    # Comparison plots
    similarity_plot: np.ndarray
    """Aligned profiles overlaid (GetAlignedProfilesImage)."""

    comparison_overview: np.ndarray
    """Main NFI results overview figure (NfiFigureProfile)."""

    # Filtered mark images
    mark1_filtered_preview_image: np.ndarray
    """Filtered reference mark preview (after equalization)."""

    mark2_filtered_preview_image: np.ndarray
    """Filtered compared mark preview (after equalization)."""

    # Side by side and wavelength plots
    mark1_vs_moved_mark2: np.ndarray
    """Both marks side by side with gap between them."""

    wavelength_plot: np.ndarray
    """Profiles + wavelength-dependent cross-correlation plot."""


# =============================================================================
# Helper Functions (plot onto existing axes)
# =============================================================================


def _figure_to_array(fig: Figure) -> np.ndarray:
    """
    Convert a matplotlib figure to a numpy array.

    :param fig: Matplotlib figure to convert.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr[:, :, :3].copy()


def _get_fig_dimensions(height: int, width: int) -> tuple[float, float]:
    """Calculate figure dimensions based on data aspect ratio."""
    aspect_ratio = width / height
    fig_width = 10
    fig_height = fig_width / aspect_ratio
    return fig_height + 1.0, fig_width


def _plot_profiles_on_axes(
    ax: plt.Axes,
    profile_ref: NDArray,
    profile_comp: NDArray,
    scale: float,
    score: float,
    title_prefix: str = "Similarity Score (Correlation Coefficient)",
) -> None:
    """
    Plot two aligned profiles on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param profile_ref: Reference profile (aligned).
    :param profile_comp: Compared profile (aligned).
    :param scale: scale of the profiles in meters.
    :param score: Pre-computed correlation coefficient.
    :param title_prefix: Prefix for the title before the correlation value.
    """
    profile1 = profile_ref.flatten()
    profile2 = profile_comp.flatten()

    x1 = np.arange(len(profile1)) * scale * 1e6  # µm
    x2 = np.arange(len(profile2)) * scale * 1e6

    y1 = profile1 * 1e6  # µm
    y2 = profile2 * 1e6

    ax.plot(x1, y1, "b-", label="Reference Profile A", linewidth=1.5)
    ax.plot(x2, y2, "r-", label="Compared Profile B", linewidth=1.5)

    ax.set_xlabel("Profile Length [µm]", fontsize=11)
    ax.set_ylabel("Profile Height [µm]", fontsize=11)
    ax.set_title(f"{title_prefix}: {score:.5f}", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_depthmap_on_axes(
    ax: plt.Axes,
    fig: Figure,
    data: NDArray,
    scale: float,
    title: str,
    shrink_colorbar: float = 1.0,
) -> None:
    """
    Plot a depth map on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param data: Data to plot in meters.
    :param scale: Scale of the data in meters.
    :param title: Title for the plot.
    :param shrink_colorbar: Shrink factor for colorbar (0-1).
    """
    height, width = data.shape
    extent = [0, width * scale * 1e6, 0, height * scale * 1e6]

    im = ax.imshow(
        data * 1e6,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin="lower",
        extent=extent,
    )
    ax.set_xlabel("X - Position [µm]", fontsize=11)
    ax.set_ylabel("Y - Position [µm]", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    cbar = fig.colorbar(im, ax=ax, label="µm", shrink=shrink_colorbar)
    cbar.ax.tick_params(labelsize=10)


def _plot_side_by_side_on_axes(
    ax: plt.Axes,
    fig: Figure,
    data_ref: NDArray,
    data_comp: NDArray,
    scale: float,
    title: str = "Reference Surface A / Moved Compared Surface B",
    shrink_colorbar: float = 1.0,
) -> None:
    """
    Plot two surfaces side by side on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param data_ref: Reference data in meters.
    :param data_comp: Compared data in meters.
    :param scale: Scale of the data in meters.
    :param title: Title for the plot.
    :param shrink_colorbar: Shrink factor for colorbar (0-1).
    """
    gap_width = int(np.ceil(min(data_ref.shape[1], data_comp.shape[1]) / 100))
    gap = np.full((data_ref.shape[0], gap_width), np.nan)
    combined = np.hstack([data_ref, gap, data_comp])

    _plot_depthmap_on_axes(ax, fig, combined, scale, title, shrink_colorbar)


# =============================================================================
# Public Plotting Functions
# =============================================================================


def plot_similarity(
    profile_ref: NDArray,
    profile_comp: NDArray,
    scale: float,
    score: float,
) -> np.ndarray:
    """
    Plot two aligned profiles overlaid (similarity plot).

    :param profile_ref: Reference profile (aligned).
    :param profile_comp: Compared profile (aligned).
    :param scale: scale of the profiles in meters.
    :param score: Pre-computed correlation coefficient from ProfileCorrelatorSingle.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_PROFILES)

    _plot_profiles_on_axes(ax, profile_ref, profile_comp, scale, score)
    fig.tight_layout(h_pad=3.0)
    arr = _figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_wavelength_xcorr(
    profile_ref: NDArray,
    profile_comp: NDArray,
    scale: float,
    xcorr_data: WavelengthXcorrData,
) -> WavelengthXcorrResult:
    """
    Plot aligned profiles with wavelength-range dependent cross-correlation.

    :param profile_ref: Reference profile (aligned).
    :param profile_comp: Compared profile (aligned).
    :param scale: scale of the profiles in meters.
    :param xcorr_data: Pre-computed wavelength-dependent correlation data.
    :returns: WavelengthXcorrResult with rendered image.
    """
    fig, axes = plt.subplots(2, 1, figsize=DEFAULT_FIGSIZE_PROFILES_XCORR)

    # === Subplot 1: Aligned profiles ===
    _plot_profiles_on_axes(
        axes[0],
        profile_ref,
        profile_comp,
        scale,
        xcorr_data.xcorr_full_range,
        title_prefix="Similarity Score (Cross Coefficient), full range",
    )

    # === Subplot 2: Wavelength-dependent correlation ===
    ax2 = axes[1]

    passbands = xcorr_data.quality_passbands
    correlations = xcorr_data.range_dependent_xcorrs
    xx = np.arange(1, len(passbands) + 1)

    if len(passbands) > 1:
        ax2.plot(xx, correlations * 100, "b-*", linewidth=2)
    else:
        ax2.plot(xx, correlations * 100, "b*", markersize=10)

    ax2.set_ylim(-5, 105)
    ax2.set_xlim(xx[0] - xx[-1] * 0.01, xx[-1] + xx[-1] * 0.01)
    ax2.grid(True)
    ax2.tick_params(labelsize=14)

    # X tick labels showing wavelength ranges
    labels = [f"{int(lo)}-{int(hi)}" for lo, hi in passbands]
    ax2.set_xticks(xx)
    ax2.set_xticklabels(labels)

    ax2.set_xlabel("Wavelength ranges of geometrical details [µm]", fontsize=14)
    ax2.set_ylabel("Correlation Coefficient", fontsize=14)

    fig.tight_layout(h_pad=3.0)
    arr = _figure_to_array(fig)
    plt.close(fig)

    return WavelengthXcorrResult(image=arr, data=xcorr_data)


def plot_depthmap_with_axes(data: NDArray, scale: float, title: str) -> np.ndarray:
    """
    Plot a depth map rendering of a mark.

    :param data: data to plot in meters.
    :param scale: scale of the data in meters.
    :param title: Title for the plot.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    height, width = data.shape
    fig_height, fig_width = _get_fig_dimensions(height, width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    _plot_depthmap_on_axes(ax, fig, data, scale, title, shrink_colorbar=0.5)

    fig.tight_layout()
    arr = _figure_to_array(fig)
    plt.close(fig)
    return arr


def plot_side_by_side_surfaces(
    data_ref: NDArray,
    data_comp: NDArray,
    scale: float,
) -> np.ndarray:
    """
    Plot two aligned marks side by side with a small gap.

    :param data_ref: Reference data (aligned) in meters.
    :param data_comp: Compared data (aligned) in meters.
    :param scale: Scale of the data in meters.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    # Create combined data for size calculation
    gap_width = int(np.ceil(min(data_ref.shape[1], data_comp.shape[1]) / 100))
    combined_width = data_ref.shape[1] + gap_width + data_comp.shape[1]
    height = data_ref.shape[0]

    fig_height, fig_width = _get_fig_dimensions(height, combined_width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    _plot_side_by_side_on_axes(ax, fig, data_ref, data_comp, scale, shrink_colorbar=0.5)

    fig.tight_layout()
    arr = _figure_to_array(fig)
    plt.close(fig)
    return arr


def _draw_metadata_box(ax: plt.Axes, title: str, metadata: dict) -> None:
    """Draw a metadata box with title and key-value pairs."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    # Title above box
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Create table data with text wrapping
    table_data = []
    wrap_width = 22  # characters before wrapping

    for k, v in metadata.items():
        wrapped_lines = textwrap.wrap(str(v), width=wrap_width)
        if not wrapped_lines:
            wrapped_lines = [""]

        # First line has the key
        table_data.append([f"{k}:", wrapped_lines[0]])

        # Continuation lines have empty key
        for line in wrapped_lines[1:]:
            table_data.append(["", line])

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        colWidths=[0.35, 0.55],
        loc="center left",
        edges="open",
        bbox=[0.07, 0.1, 0.9, 0.8],  # [left, bottom, width, height]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)  # Scale row height

    # Make keys bold
    for i in range(len(table_data)):
        table[i, 0].set_text_props(fontweight="bold", ha="right")
        table[i, 1].set_text_props(ha="left")


def _draw_results_box(ax: plt.Axes, metrics: CorrelationMetrics) -> None:
    """Draw the results metrics box."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Build metrics as key-value pairs
    items = []
    wrap_width = 25  # characters before wrapping

    def add_item(key: str, value: str) -> None:
        """Add an item with text wrapping for long values."""
        wrapped_lines = textwrap.wrap(str(value), width=wrap_width)
        if not wrapped_lines:
            wrapped_lines = [""]
        items.append((key, wrapped_lines[0]))
        for line in wrapped_lines[1:]:
            items.append(("", line))

    if metrics.date_report:
        add_item("Date report:", metrics.date_report)
    if metrics.mark_type:
        add_item("Mark type:", metrics.mark_type)

    add_item("Correlation Coefficient:", f"{metrics.score:.4f}")

    if metrics.sq_a is not None:
        add_item("Sq(A):", f"{metrics.sq_a:.4f} µm")
    if metrics.sq_b is not None:
        add_item("Sq(B):", f"{metrics.sq_b:.4f} µm")
    if metrics.sq_b_minus_a is not None:
        add_item("Sq(B-A):", f"{metrics.sq_b_minus_a:.4f} µm")
    if metrics.sq_ratio is not None:
        add_item("Sq(B) / Sq(A):", f"{metrics.sq_ratio:.4f} %")
    if metrics.sign_diff_dsab is not None:
        add_item("Sign. Diff. DsAB:", f"{metrics.sign_diff_dsab:.2f} %")

    add_item("Overlap:", f"{metrics.overlap:.2f} %")

    if metrics.data_spacing is not None:
        items.append(("", ""))  # blank line
        add_item("Data spacing:", f"{metrics.data_spacing:.4f} µm")
    if metrics.cutoff_low_pass is not None:
        add_item("Cutoff length low-pass filter:", f"{metrics.cutoff_low_pass:.0f} µm")
    if metrics.cutoff_high_pass is not None:
        add_item(
            "Cutoff length high-pass filter:", f"{metrics.cutoff_high_pass:.0f} µm"
        )

    # Create table data
    table_data = [[k, v] for k, v in items]

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        colWidths=[0.55, 0.45],
        loc="upper center",
        edges="open",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)  # Scale row height

    # Make keys bold and right-aligned
    for i in range(len(table_data)):
        table[i, 0].set_text_props(fontweight="bold", ha="right")
        table[i, 1].set_text_props(ha="left")


def plot_comparison_overview(
    mark_ref: Mark,
    mark_comp: Mark,
    mark_ref_aligned: Mark,
    mark_comp_aligned: Mark,
    profile_ref: Mark,
    profile_comp: Mark,
    metrics: CorrelationMetrics,
    metadata_ref: dict,
    metadata_comp: dict,
) -> np.ndarray:
    """
    Generate the main NFI results overview figure.

    This creates a composite figure with:
    - Row 0: Metadata tables for both profiles (with boxes)
    - Row 1: Filtered surface depth maps + Results metrics
    - Row 2: Side-by-side aligned surfaces
    - Row 3: Profile comparison plot

    :param mark_ref: Reference mark (filtered).
    :param mark_comp: Compared mark (filtered).
    :param mark_ref_aligned: Reference mark after alignment.
    :param mark_comp_aligned: Compared mark after alignment.
    :param profile_ref: Reference profile (aligned).
    :param profile_comp: Compared profile (aligned).
    :param metrics: Correlation metrics to display.
    :param metadata_ref: Metadata dict for reference profile.
    :param metadata_comp: Metadata dict for compared profile.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    fig = plt.figure(figsize=DEFAULT_FIGSIZE_OVERVIEW)

    # Create grid layout: 4 rows, 3 columns
    # Row 0: metadata boxes (2 columns)
    # Row 1: filtered surfaces (2 columns) + results (1 column)
    # Row 2: side-by-side (2 columns)
    # Row 3: profile plot (3 columns)
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[0.18, 0.32, 0.25, 0.25],
        width_ratios=[0.35, 0.35, 0.30],
        hspace=0.4,
        wspace=0.3,
    )

    # === Row 0: Metadata tables with boxes ===
    ax_meta_ref = fig.add_subplot(gs[0, 0])
    _draw_metadata_box(ax_meta_ref, "Reference Profile (A)", metadata_ref)

    ax_meta_comp = fig.add_subplot(gs[0, 1])
    _draw_metadata_box(ax_meta_comp, "Compared Profile (B)", metadata_comp)

    scale = mark_ref.scan_image.scale_x

    # === Row 1: Filtered surface depth maps + Results ===
    ax_ref = fig.add_subplot(gs[1, 0])
    _plot_depthmap_on_axes(
        ax_ref,
        fig,
        mark_ref.scan_image.data,
        scale,
        title="Filtered Reference Surface A",
    )

    ax_comp = fig.add_subplot(gs[1, 1])
    _plot_depthmap_on_axes(
        ax_comp,
        fig,
        mark_comp.scan_image.data,
        scale,
        title="Filtered Compared Surface B",
    )

    # Results metrics to the right
    ax_results = fig.add_subplot(gs[1, 2])
    _draw_results_box(ax_results, metrics)

    # === Row 2: Side-by-side ===
    ax_side = fig.add_subplot(gs[2, :2])
    _plot_side_by_side_on_axes(
        ax_side,
        fig,
        mark_ref_aligned.scan_image.data,
        mark_comp_aligned.scan_image.data,
        scale,
    )

    # === Row 3: Profile plot ===
    ax_profile = fig.add_subplot(gs[3, :])
    _plot_profiles_on_axes(
        ax_profile,
        profile_ref.scan_image.data,
        profile_comp.scan_image.data,
        profile_ref.scan_image.scale_x,
        metrics.score,
        title_prefix="Reference Profile A / Moved Compared Profile B",
    )

    fig.tight_layout()
    arr = _figure_to_array(fig)
    plt.close(fig)
    return arr


# =============================================================================
# Main Entry Point
# =============================================================================


def plot_striation_comparison_results(
    mark_ref_filtered: Mark,
    mark_comp_filtered: Mark,
    mark_ref_aligned: Mark,
    mark_comp_aligned: Mark,
    profile_ref_aligned: Mark,
    profile_comp_aligned: Mark,
    metrics: CorrelationMetrics,
    xcorr_data: WavelengthXcorrData,
    metadata_ref: dict,
    metadata_comp: dict,
) -> StriationComparisonResults:
    """
    Generate visualization results for striation (profile) correlation comparison.

    :param mark_ref_filtered: Reference mark after filtering/equalization.
    :param mark_comp_filtered: Compared mark after filtering/equalization.
    :param mark_ref_aligned: Reference mark after alignment (for side-by-side).
    :param mark_comp_aligned: Compared mark after alignment (for side-by-side).
    :param profile_ref_aligned: Reference profile after alignment.
    :param profile_comp_aligned: Compared profile after alignment.
    :param metrics: Correlation metrics to display in overview.
    :param xcorr_data: Pre-computed wavelength-dependent correlation data.
    :param metadata_ref: Metadata dict for reference profile display.
    :param metadata_comp: Metadata dict for compared profile display.
    :returns: StriationComparisonResults with all rendered images.
    """
    # Filtered mark images (with axes and colorbar)
    mark1_filtered_preview_image = plot_depthmap_with_axes(
        data=mark_ref_filtered.scan_image.data,
        scale=mark_ref_filtered.scan_image.scale_x,
        title="Filtered Reference Surface A",
    )

    mark2_filtered_preview_image = plot_depthmap_with_axes(
        data=mark_comp_filtered.scan_image.data,
        scale=mark_comp_filtered.scan_image.scale_x,
        title="Filtered Compared Surface B",
    )

    # Comparison overview
    comparison_overview = plot_comparison_overview(
        mark_ref=mark_ref_filtered,
        mark_comp=mark_comp_filtered,
        mark_ref_aligned=mark_ref_aligned,
        mark_comp_aligned=mark_comp_aligned,
        profile_ref=profile_ref_aligned,
        profile_comp=profile_comp_aligned,
        metrics=metrics,
        metadata_ref=metadata_ref,
        metadata_comp=metadata_comp,
    )

    # Side by side
    mark1_vs_moved_mark2 = plot_side_by_side_surfaces(
        data_ref=mark_ref_aligned.scan_image.data,
        data_comp=mark_comp_aligned.scan_image.data,
        scale=mark_ref_aligned.scan_image.scale_x,
    )

    # Profile plots
    similarity_plot = plot_similarity(
        profile_ref=profile_ref_aligned.scan_image.data,
        profile_comp=profile_comp_aligned.scan_image.data,
        scale=profile_ref_aligned.scan_image.scale_x,
        score=metrics.score,
    )

    xcorr_result = plot_wavelength_xcorr(
        profile_ref=profile_ref_aligned.scan_image.data,
        profile_comp=profile_comp_aligned.scan_image.data,
        scale=profile_ref_aligned.scan_image.scale_x,
        xcorr_data=xcorr_data,
    )

    return StriationComparisonResults(
        similarity_plot=similarity_plot,
        comparison_overview=comparison_overview,
        mark1_filtered_preview_image=mark1_filtered_preview_image,
        mark2_filtered_preview_image=mark2_filtered_preview_image,
        mark1_vs_moved_mark2=mark1_vs_moved_mark2,
        wavelength_plot=xcorr_result.image,
    )
