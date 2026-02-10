import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import Any
from container_models.base import FloatArray1D


def plot_score_histograms_kde(
    scores: FloatArray1D,
    y: FloatArray1D,
    new_score: float | None = None,
    bins: int | None = 20,
    ax: Any = plt,
    show_density: bool = True,
) -> None:
    """
    Create score histograms with kernel density estimates for KM and KNM datasets.

    Parameters:
    -----------
    scores : FloatArray1D
        Array of score values
    y : FloatArray1D
        Array of labels (0 for KNM, 1 for KM)
    new_score : float, optional
        A new score value to plot as a green vertical line
    bins : int, optional
        Number of bins for histogram. If None, uses 'auto' binning.
    ax : matplotlib axis, optional
        The matplotlib axis to plot on (default: plt)
    show_density : bool, optional
        Whether to show the kernel density estimate curves (default: True)
    """

    # Separate data by label
    knm_scores = scores[y == 0]  # KNM (label 0)
    km_scores = scores[y == 1]  # KM (label 1)

    # Set up bin edges
    max_score = scores.max() * 1.05
    if bins:
        bin_edges = np.linspace(0, max_score, bins + 1)
    else:
        bin_edges = np.histogram_bin_edges(scores, bins="auto")

    # Plot histograms
    ax.hist(
        knm_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="blue",
        label=f"KNM (n={len(knm_scores)})",
    )

    ax.hist(
        km_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="orange",
        label=f"KM (n={len(km_scores)})",
    )

    # Plot KDE curves if requested
    if show_density:
        x = np.linspace(bin_edges[0], bin_edges[-1], 500)

        if len(knm_scores) > 1:
            kde_knm = gaussian_kde(knm_scores)
            ax.plot(x, kde_knm(x), color="blue", linestyle="--", linewidth=2)

        if len(km_scores) > 1:
            kde_km = gaussian_kde(km_scores)
            ax.plot(x, kde_km(x), color="orange", linestyle="--", linewidth=2)

    # Add vertical line for new score if provided
    if new_score is not None:
        ax.axvline(
            x=new_score,
            color="green",
            linestyle="-",
            linewidth=2.5,
            zorder=10,
            label=f"new score ({new_score:.2f})",
        )

    # Set labels and formatting
    # Handle both plt module and Axes object
    if hasattr(ax, "set_xlabel"):
        # ax is an Axes object
        ax.set_xlabel("Score")
        ax.set_ylabel("Normalized density")
        ax.set_xlim(0, max_score)
        ax.set_ylim(0, None)
    else:
        # ax is plt module
        ax.xlabel("Score")
        ax.ylabel("Normalized density")
        ax.xlim(0, max_score)
        ax.ylim(0, None)

    ax.legend()
    ax.grid(True, linestyle=":")


# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)

    # KNM data (n=171991) - concentrated near 0
    n_knm = 171991
    knm_scores_data = np.random.gamma(0.5, 0.5, n_knm)

    # KM data (n=1125) - more spread out
    n_km = 1125
    n_km_low = 787  # 70% of 1125
    n_km_high = 338  # 30% of 1125

    km_scores_data = np.concatenate(
        [
            np.random.gamma(1.5, 2, n_km_low),
            np.random.uniform(10, 50, n_km_high),
        ]
    )

    # Combine into numpy arrays
    scores = np.concatenate([knm_scores_data, km_scores_data])
    y = np.concatenate([np.zeros(n_knm), np.ones(n_km)])

    # Define a new score to highlight
    new_score = 5.0

    # Example 1: Using default plt (creates new figure)
    plt.figure(figsize=(10, 6))
    plot_score_histograms_kde(scores, y, new_score=new_score, bins=50)
    plt.title("Score histograms with KDE")
    plt.savefig(
        "/mnt/user-data/outputs/score_histograms_example1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✓ Example 1 saved: score_histograms_example1.png")

    # Example 2: Using custom axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_score_histograms_kde(scores, y, new_score=new_score, bins=50, ax=ax)
    ax.set_title("Score histograms with KDE (custom axis)")
    plt.savefig(
        "/mnt/user-data/outputs/score_histograms_example2.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✓ Example 2 saved: score_histograms_example2.png")

    # Example 3: Without density curves
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_score_histograms_kde(
        scores, y, new_score=None, bins=50, ax=ax, show_density=False
    )
    ax.set_title("Score histograms without KDE")
    plt.savefig(
        "/mnt/user-data/outputs/score_histograms_example3.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✓ Example 3 saved: score_histograms_example3.png")

    print("\nAll examples completed successfully!")
