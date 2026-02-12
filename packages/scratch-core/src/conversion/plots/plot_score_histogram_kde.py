import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.axes import Axes
from container_models.base import FloatArray1D


def plot_score_histograms_kde(
    scores: FloatArray1D,
    y: FloatArray1D,
    ax: Axes,
    new_score: float | None = None,
    bins: int | None = 20,
    show_density: bool = True,
    bandwidth: float | str | None = "silverman",
) -> None:
    """
    Create score histograms with optional new_score line and kernel density estimates for KM and KNM datasets.

    Parameters:
    -----------
    :param scores: FloatArray1D
        Array of score values
    :param y: FloatArray1D
        Array of labels (0 for KNM, 1 for KM)
    :param ax : matplotlib.axes.Axes
        The axis to plot on
    :param new_score : float, optional
        A new score value to plot as a vertical line
    :param bins : int, optional
        Number of bins for histogram. If None, uses 'auto' binning.
    :param show_density : bool, optional
        Whether to show the kernel density estimate curves
    :param bandwidth : float | {'silverman', 'scott'} | None
        KDE bandwidth method or value, None defaults to 'scott'
    """

    if ax is None:
        raise ValueError(
            "ax must be a matplotlib.axes.Axes object, created by e.g. 'fig, axes = plt.subplots()'"
        )

    if isinstance(bandwidth, str) and bandwidth not in {"silverman", "scott"}:
        raise TypeError("bandwidth must be a float, 'silverman', 'scott', or None")

    # Separate data by label
    knm_scores = scores[y == 0]
    km_scores = scores[y == 1]

    # Bin edges
    max_score = scores.max() * 1.05
    if bins is not None:
        bin_edges = np.linspace(0, max_score, bins + 1)
    else:
        bin_edges = np.histogram_bin_edges(scores, bins="auto")
    bin_edges = list(bin_edges)

    # Histograms
    hist_knm = ax.hist(
        knm_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="blue",
        label=f"KNM (n={len(knm_scores)})",
    )

    hist_km = ax.hist(
        km_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="orange",
        label=f"KM (n={len(km_scores)})",
    )

    # Y-limit scaling
    max_y = np.max(np.concatenate([hist_knm[0], hist_km[0]])) * 1.1

    # KDE
    if show_density:
        x = np.linspace(0, bin_edges[-1], 500)

        if len(knm_scores) > 1:
            kde_knm = gaussian_kde(knm_scores, bw_method=bandwidth)
            ax.plot(x, kde_knm(x), color="blue", linestyle="--", linewidth=2)

        if len(km_scores) > 1:
            kde_km = gaussian_kde(km_scores, bw_method=bandwidth)
            ax.plot(x, kde_km(x), color="orange", linestyle="--", linewidth=2)

    # Vertical line for new_score
    if new_score is not None:
        ax.axvline(
            x=new_score,
            color="green",
            linestyle="-",
            linewidth=2.5,
            zorder=10,
            label=f"new score ({new_score:.2f})",
        )

    # Labels & formatting
    ax.set_xlabel("Score")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(0, max_score)
    ax.set_ylim(0, max_y)
    ax.legend()
    ax.grid(True, linestyle=":")
