import numpy as np
from matplotlib.axes import Axes

from conversion.plots.data_formats import HistogramData


def plot_score_histograms(ax: Axes, data: HistogramData) -> None:
    """
    Create score histograms with optional new_score line and kernel density estimates for KM and KNM datasets.

    :param ax: The axis to plot on.
    :param data: Histogram input data containing scores, labels, bins, densities, and new_score.
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
            data.densities["x"],
            data.densities["km_density_at_x"],
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
            data.densities["x"],
            data.densities["knm_density_at_x"],
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
    ax.set_xlabel("Score")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(0, max_score)
    ax.set_ylim(0, max_y)
    ax.legend()
    ax.grid(True, linestyle=":")
