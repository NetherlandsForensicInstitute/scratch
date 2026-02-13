"""
Module for creating complete CCF comparison analysis plots.

This module combines striation comparison plots (heatmaps, profiles, metadata)
with score distribution analysis (histograms and LogLR transformation) into
a single comprehensive figure matching the reference layout.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.figure import Figure

from container_models.base import FloatArray1D, FloatArray2D
from conversion.plots.plot_score_histograms import DensityDict, plot_score_histograms
from conversion.plots.plot_score_llr_transformation import plot_score_llr_transformation


def plot_ccf_comparison_complete(
    # Metadata for top row
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
    metadata_results: dict[str, str],
    # Heatmap data for row 2
    data_reference_filtered: FloatArray2D,
    data_compared_filtered: FloatArray2D,
    scale_heatmap: float,
    # Side-by-side data for row 3
    data_reference_aligned: FloatArray2D,
    data_compared_aligned: FloatArray2D,
    # Score distribution data for row 4
    scores: FloatArray1D,
    labels: FloatArray1D,
    scores_transformed: FloatArray1D,
    llrs: FloatArray1D,
    llrs_at5: FloatArray1D,
    llrs_at95: FloatArray1D,
    # Optional parameters
    bins: int | None = None,
    densities: DensityDict | None = None,
    densities_transformed: DensityDict | None = None,
    new_score: float | None = None,
    score_llr_point: tuple[float, float] | None = None,
) -> Figure:
    """
    Create complete CCF comparison analysis figure.

    The figure layout (3 rows):
    - Row 0: Metadata tables (Reference Profile A | Compared Profile B | empty)
    - Row 1: Filtered Reference Surface A | Filtered Compared Surface B | Side-by-side + Results
    - Row 2: Score analysis (Score histograms | Transformed histograms | LogLR plot)

    :param metadata_reference: Metadata dictionary for reference profile
    :param metadata_compared: Metadata dictionary for compared profile
    :param metadata_results: Results/metrics dictionary
    :param data_reference_filtered: Filtered reference surface data (2D array)
    :param data_compared_filtered: Filtered compared surface data (2D array)
    :param scale_heatmap: Scale for heatmaps in meters
    :param data_reference_aligned: Aligned reference data for side-by-side (2D array)
    :param data_compared_aligned: Aligned compared data for side-by-side (2D array)
    :param scores: Original score values (1D array)
    :param labels: Labels for scores (0=KNM, 1=KM, 1D array)
    :param scores_transformed: Transformed score values (1D array)
    :param llrs: Log-likelihood ratios (1D array)
    :param llrs_at5: LLRs at 5% confidence (1D array)
    :param llrs_at95: LLRs at 95% confidence (1D array)
    :param bins: Number of bins for histograms (optional)
    :param densities: Density estimates for original scores (optional)
    :param densities_transformed: Density estimates for transformed scores (optional)
    :param new_score: Score value to mark with vertical line (optional)
    :param score_llr_point: (score, llr) coordinate for crosshairs (optional)
    :return: Matplotlib Figure object
    :raises ValueError: If array dimensions don't match

    Example::

        >>> import numpy as np
        >>> # Create sample data
        >>> metadata_ref = {"Case ID": "2022_07_21_126", "Firearm ID": "unknown_firearm_4"}
        >>> metadata_comp = {"Case ID": "2022_07_21_126", "Firearm ID": "unknown_firearm_7"}
        >>> metadata_results = {"Date report": "2023-02-16", "Score": "0.97 (1.86)"}
        >>> data_ref = np.random.randn(100, 100)
        >>> data_comp = np.random.randn(100, 100)
        >>> scores = np.random.beta(2, 5, 1000)
        >>> labels = np.random.randint(0, 2, 1000)
        >>> fig = plot_ccf_comparison_complete(
        ...     metadata_ref, metadata_comp, metadata_results,
        ...     data_ref, data_comp, 1e-6,
        ...     data_ref, data_comp,
        ...     scores, labels, scores*0.5+0.5,
        ...     np.random.randn(100), np.random.randn(100), np.random.randn(100)
        ... )
    """
    # Validate inputs
    if len(scores) != len(labels):
        raise ValueError(
            f"Length mismatch: scores ({len(scores)}) vs labels ({len(labels)})"
        )

    if len(llrs) != len(llrs_at5) or len(llrs) != len(llrs_at95):
        raise ValueError(
            f"LLR array length mismatch: llrs ({len(llrs)}), "
            f"llrs_at5 ({len(llrs_at5)}), llrs_at95 ({len(llrs_at95)})"
        )

    # Create figure with appropriate size
    fig = plt.figure(figsize=(16, 15))

    # Define grid: 3 rows with varying heights
    # Row 0: Metadata (smaller)
    # Row 1: Heatmaps + Side-by-side + Results (medium)
    # Row 2: Score distributions (larger)
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[0.12, 0.38, 0.50],
        hspace=0.35,
        wspace=0.25,
    )

    # ========== ROW 0: METADATA TABLES ==========
    # Reference Profile (A)
    ax_meta_ref = fig.add_subplot(gs[0, 0])
    _draw_metadata_table(ax_meta_ref, metadata_reference, "Reference Profile (A)")

    # Compared Profile (B)
    ax_meta_comp = fig.add_subplot(gs[0, 1])
    _draw_metadata_table(ax_meta_comp, metadata_compared, "Compared Profile (B)")

    # Empty space (or could add summary info)
    ax_meta_empty = fig.add_subplot(gs[0, 2])
    ax_meta_empty.axis("off")

    # ========== ROW 1: HEATMAPS + SIDE-BY-SIDE + RESULTS ==========
    # Filtered Reference Surface A
    ax_heatmap_ref = fig.add_subplot(gs[1, 0])
    _plot_simple_heatmap(
        ax_heatmap_ref,
        data_reference_filtered,
        scale_heatmap,
        "Filtered Reference Surface A",
    )

    # Filtered Compared Surface B
    ax_heatmap_comp = fig.add_subplot(gs[1, 1])
    _plot_simple_heatmap(
        ax_heatmap_comp,
        data_compared_filtered,
        scale_heatmap,
        "Filtered Compared Surface B",
    )

    # Side-by-side + Results combined
    # Split the third column into top (side-by-side) and bottom (results)
    gs_col3 = gs[1, 2].subgridspec(2, 1, hspace=0.3, height_ratios=[0.65, 0.35])

    ax_side = fig.add_subplot(gs_col3[0])
    _plot_side_by_side(
        ax_side, data_reference_aligned, data_compared_aligned, scale_heatmap
    )

    ax_results = fig.add_subplot(gs_col3[1])
    _draw_metadata_table(ax_results, metadata_results, None, draw_border=False)

    # ========== ROW 2: SCORE DISTRIBUTIONS ==========
    # Create score grid for LLR if needed
    if len(llrs) != len(scores_transformed):
        score_grid = np.linspace(
            scores_transformed.min(), scores_transformed.max(), len(llrs)
        )
    else:
        score_grid = scores_transformed

    # Left: Score histograms
    ax_hist = fig.add_subplot(gs[2, 0])
    plot_score_histograms(
        scores=scores,
        labels=labels,
        ax=ax_hist,
        bins=bins,
        densities=densities,
        new_score=new_score,
    )
    ax_hist.set_title("Score histograms", fontsize=12, fontweight="bold")

    # Middle: Transformed score histograms
    ax_hist_trans = fig.add_subplot(gs[2, 1])
    plot_score_histograms(
        scores=scores_transformed,
        labels=labels,
        ax=ax_hist_trans,
        bins=bins,
        densities=densities_transformed,
        new_score=None,
    )
    ax_hist_trans.set_title(
        "Transformed score histograms", fontsize=12, fontweight="bold"
    )

    # Right: LogLR plot
    ax_llr = fig.add_subplot(gs[2, 2])
    plot_score_llr_transformation(
        ax=ax_llr,
        scores=score_grid,
        llrs=llrs,
        llrs_at5=llrs_at5,
        llrs_at95=llrs_at95,
        score_llr_point=score_llr_point,
    )

    plt.tight_layout()
    return fig


def _draw_metadata_table(
    ax, metadata: dict[str, str], title: str | None = None, draw_border: bool = True
):
    """
    Draw a metadata table on an axis.

    :param ax: Matplotlib axes
    :param metadata: Dictionary of key-value pairs
    :param title: Optional title for the table
    :param draw_border: Whether to draw border around table
    """
    ax.axis("off")

    if title:
        ax.text(
            0.5,
            0.95,
            title,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
        )
        y_start = 0.85
    else:
        y_start = 0.95

    # Draw metadata items
    y_pos = y_start
    y_step = 0.15 if len(metadata) <= 5 else 0.12

    for key, value in metadata.items():
        ax.text(
            0.05,
            y_pos,
            f"{key}:",
            ha="left",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
        )
        ax.text(
            0.98,
            y_pos,
            str(value),
            ha="right",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
        )
        y_pos -= y_step

    if draw_border:
        # Draw border
        from matplotlib.patches import Rectangle

        border = Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(border)


def _plot_simple_heatmap(ax, data: FloatArray2D, scale: float, title: str):
    """
    Plot a simple heatmap on an axis.

    :param ax: Matplotlib axes
    :param data: 2D array of data
    :param scale: Scale in meters
    :param title: Title for the plot
    """
    # Convert scale to micrometers for display
    scale_um = scale * 1e6

    extent = [
        0,
        data.shape[1] * scale_um,
        0,
        data.shape[0] * scale_um,
    ]

    # Convert data to micrometers
    data_um = data * 1e6

    im = ax.imshow(
        data_um,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )

    ax.set_xlabel("X - Position [μm]", fontsize=10)
    ax.set_ylabel("Y - Position [μm]", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("μm", fontsize=9)


def _plot_side_by_side(
    ax, data_reference: FloatArray2D, data_compared: FloatArray2D, scale: float
):
    """
    Plot two surfaces side by side with a gap.

    :param ax: Matplotlib axes
    :param data_reference: Reference data (2D array)
    :param data_compared: Compared data (2D array)
    :param scale: Scale in meters
    """
    # Create gap
    gap_width = int(np.ceil(min(data_reference.shape[1], data_compared.shape[1]) / 100))
    gap = np.full((data_reference.shape[0], gap_width), np.nan)

    # Combine data
    combined = np.concatenate([data_reference, gap, data_compared], axis=1)

    # Convert to micrometers
    scale_um = scale * 1e6
    combined_um = combined * 1e6

    extent = [
        0,
        combined.shape[1] * scale_um,
        0,
        combined.shape[0] * scale_um,
    ]

    im = ax.imshow(
        combined_um,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )

    ax.set_xlabel("X - Position [μm]", fontsize=10)
    ax.set_ylabel("Y - Position [μm]", fontsize=10)
    ax.set_title(
        "Reference Surface A / Moved Compared Surface B", fontsize=11, fontweight="bold"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("μm", fontsize=9)
