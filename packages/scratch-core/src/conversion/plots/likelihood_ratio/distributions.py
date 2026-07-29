import numpy as np
from matplotlib.axes import Axes

from conversion.plots.likelihood_ratio.data_formats import (
    HistogramData,
    LlrTransformationData,
)


def plot_score_histograms(
    ax: Axes, data: HistogramData, title: str = "Score histograms"
) -> None:
    """
    Create score histograms with optional new_score line and kernel density estimates for KM and KNM datasets.

    :param ax: The axis to plot on.
    :param data: Histogram input data containing scores, labels, bins, densities, and new_score.
    :param title: Axes title.
    """

    # Separate data by label
    knm_scores = data.scores[data.labels == 0]
    km_scores = data.scores[data.labels == 1]

    # Bin edges
    max_score = data.scores.max() * 1.05
    if data.bins is not None:
        bin_edges = np.linspace(0, max_score, data.bins + 1)
    else:
        bin_edges = np.histogram_bin_edges(
            data.scores, range=(0, max_score), bins="auto"
        )
    bin_edges = list(bin_edges)

    # Plot things in right order for getting legend items in the right order
    # Histograms and optional densities
    barheights_km, _, _ = ax.hist(
        km_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="orange",
        label=f"KM (n={len(km_scores)})",
    )

    if data.densities:
        ax.plot(
            data.densities.x,
            data.densities.km_density_at_x,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="KM density",
        )

    barheights_knm, _, _ = ax.hist(
        knm_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="blue",
        label=f"KNM (n={len(knm_scores)})",
    )

    if data.densities:
        ax.plot(
            data.densities.x,
            data.densities.knm_density_at_x,
            color="blue",
            linestyle="--",
            linewidth=2,
            label="KNM density",
        )

    # Vertical line for new_score
    if data.new_score is not None:
        ax.axvline(
            x=data.new_score,
            color="green",
            linestyle="-",
            linewidth=2.5,
            zorder=10,
            label=f"new score ({data.new_score:.2f})",
        )

    # Y-limit scaling
    max_y = np.max(np.concatenate([barheights_knm, barheights_km])) * 1.1

    # Labels & formatting
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(0, max_score)
    ax.set_ylim(0, max_y)
    ax.legend()
    ax.grid(True, linestyle=":")


def plot_score_llr_transformation(ax: Axes, data: LlrTransformationData) -> None:
    """
    Plot LogLR curve with confidence intervals.

    :param ax: Matplotlib axes object to plot on.
    :param data: LLR transformation data containing scores, llrs, confidence intervals,
        and optional score_llr_point.
    """

    # Plot main LogLR curve
    ax.plot(data.scores, data.llrs, "b-", linewidth=2, label="LogLR all")

    # Plot confidence intervals with different dash styles
    ax.plot(
        data.scores,
        data.llrs_at5,
        "b--",
        linewidth=1,
        dashes=(5, 3),
        label="LogLR all 5%",
    )
    ax.plot(
        data.scores,
        data.llrs_at95,
        "b--",
        linewidth=1,
        dashes=(2, 2),
        label="LogLR all 95%",
    )

    # Set labels and title
    ax.set_xlabel("Score")
    ax.set_ylabel("LogLR")
    ax.set_title(
        "LogLR plot (with confidence intervals)", fontsize=12, fontweight="bold"
    )

    # Set grid
    ax.grid(True, alpha=0.3)

    # Adjust y-axis to show the full range (do this before drawing the coordinate lines)
    y_min = min(data.llrs.min(), data.llrs_at5.min())
    y_max = max(data.llrs.max(), data.llrs_at95.max())
    y_margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlim(data.scores.min(), data.scores.max())

    # Add horizontal and vertical lines at the score_llr_point
    if data.score_llr_point:
        score_point, llr_point = data.score_llr_point
        # Horizontal line from y-axis to the score point
        ax.plot(
            [data.scores.min(), score_point],
            [llr_point, llr_point],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="LogLR",
        )
        # Vertical line from x-axis to the llr point
        ax.plot(
            [score_point, score_point],
            [y_min - y_margin, llr_point],
            color="green",
            linestyle="--",
            linewidth=1.5,
        )

    # Add legend
    ax.legend(loc="upper left", frameon=True)
