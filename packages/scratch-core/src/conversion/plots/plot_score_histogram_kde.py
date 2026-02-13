import numpy as np
from matplotlib.axes import Axes
from container_models.base import FloatArray1D


def plot_score_histograms_kde(
    scores: FloatArray1D,
    labels: FloatArray1D,
    ax: Axes,
    bins: int | None,
    densities: dict[str : np.ndarray] | None,
    bandwidth: float | str | None = "silverman",
    new_score: float | None = None,
) -> None:
    """
    Create score histograms with optional new_score line and kernel density estimates for KM and KNM datasets.

    :param scores: FloatArray1D
        Array of score values
    :param labels: FloatArray1D
        Array of labels (0 for KNM, 1 for KM)
    :param ax : matplotlib.axes.Axes
        The axis to plot on
    :param bins : int, optional
        Number of bins for histogram. If None, uses 'auto' binning.
    :param densities : mapping with three keys: ('km', 'knm', 'score'), each item connects the key to a numpy array.
        This gives plot coordinates score -> density(score| H), optional
    :param bandwidth : float | {'silverman', 'scott'} | None
        KDE bandwidth method or value, None defaults to 'scott'
    :param new_score : float, optional
        A new score value to plot as a vertical line
    """

    if isinstance(bandwidth, str) and bandwidth not in {"silverman", "scott"}:
        raise ValueError("bandwidth must be a float, 'silverman', 'scott', or None")

    if densities:
        # Validate required keys
        required_keys = {"x", "km", "knm"}
        missing_keys = required_keys - set(densities.keys())

        if missing_keys:
            raise ValueError(
                f"Missing required keys in data dict: {missing_keys}. "
                f"Required keys are: {required_keys}"
            )

    # Separate data by label
    knm_scores = scores[labels == 0]
    km_scores = scores[labels == 1]

    # Bin edges
    max_score = scores.max() * 1.05
    if bins is not None:
        bin_edges = np.linspace(0, max_score, bins + 1)
    else:
        bin_edges = np.histogram_bin_edges(scores, range=(0, max_score), bins="auto")
    bin_edges = list(bin_edges)

    # Plot things in right order for getting legend items in the right order
    # Histograms and optional densities
    barheights_knm, _, _ = ax.hist(
        knm_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="blue",
        label=f"KNM (n={len(knm_scores)})",
    )

    if densities:
        ax.plot(
            densities["x"],
            densities["knm"],
            color="blue",
            linestyle="--",
            linewidth=2,
            label="KNM density",
        )

    barheights_km, _, _ = ax.hist(
        km_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="orange",
        label=f"KM (n={len(km_scores)})",
    )

    if densities:
        ax.plot(
            densities["x"],
            densities["km"],
            color="orange",
            linestyle="--",
            linewidth=2,
            label="KM density",
        )

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

    # Y-limit scaling
    max_y = np.max(np.concatenate([barheights_knm, barheights_km])) * 1.1

    # Labels & formatting
    ax.set_xlabel("Score")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(0, max_score)
    ax.set_ylim(0, max_y)
    ax.legend()
    ax.grid(True, linestyle=":")
