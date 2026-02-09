import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from container_models.base import (
    FloatArray1D,
)


def plot_score_histograms_kde(
    scores: FloatArray1D,
    y: FloatArray1D,
    new_score: float | None = None,
    bins: int | None = 20,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Create score histograms with kernel density estimates for KM and KNM datasets.

    Parameters:
    -----------
    scores : numpy array
        Array of score values
    y : numpy array
        Array of labels (0 for KNM, 1 for KM)
    new_score : float, optional
        A new score value to plot as a green vertical line
    bins : int
        Number of bins for histogram
    figsize : tuple
        Figure size (width, height)
    """

    # Separate data by label
    knm_scores = scores[y == 0]  # KNM (label 0)
    km_scores = scores[y == 1]  # KM (label 1)

    # Set up bin edges
    max_score = max(scores.max() * 1.05)
    bin_edges = np.linspace(0, max_score, bins + 1)

    # KDEs
    kde_km = gaussian_kde(km_scores)
    kde_knm = gaussian_kde(knm_scores)

    # Plot histograms and KDEs
    plt.figure(figsize=figsize)
    x = np.linspace(bin_edges[0], bin_edges[-1], 500)
    hist_km = plt.hist(
        km_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="blue",
        label=f"KM ({len(km_scores)}",
    )
    plt.plot(x, kde_km(x), color="blue", linestyle="--")
    hist_knm = plt.hist(
        knm_scores,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="orange",
        label=f"KNM ({len(knm_scores)}",
    )
    plt.plot(x, kde_knm(x), color="orange", linestyle="--")

    # Add vertical line for new score if provided
    if new_score is not None:
        plt.axvline(x=new_score, color="green", linestyle="-", linewidth=2.5, zorder=10)

    # Set axis limits
    max_hist = max(hist_km[0].max(), hist_knm[0].max())  # bin heights
    max_kde = max(kde_km(x).max(), kde_knm(x).max())  # KDE peaks

    max_y = max(max_hist, max_kde) * 1.05

    plt.ylim(0, max_y)
    plt.xlim(0, max_score)
    plt.xlabel("Score")
    plt.ylabel("Normalized density")
    plt.title("Score histograms and kde lines")
    plt.legend()
    plt.grid()
    plt.show()

    return


# Example usage
if __name__ == "__main__":
    # Generate example data similar to the image
    np.random.seed(42)

    # KNM data (n=171991) - concentrated near 0
    n_knm = 171991
    knm_scores_data = np.random.gamma(0.5, 0.5, n_knm)  # Highly skewed towards 0

    # KM data (n=1125) - more spread out
    n_km = 1125
    km_scores_data = np.concatenate(
        [
            np.random.gamma(1.5, 2, int(n_km * 0.7)),  # Most concentrated at low scores
            np.random.uniform(10, 50, int(n_km * 0.3)),  # Some spread across range
        ]
    )

    # Combine into numpy arrays
    scores = np.concatenate([knm_scores_data, km_scores_data])
    y = np.concatenate([np.zeros(n_knm), np.ones(n_km)])

    # Define a new score to highlight
    new_score = 5.0

    # Create plot
    fig, ax, ax2 = plot_score_histograms_kde(scores, y, new_score=new_score, bins=50)

    # Save figure
    plt.savefig(
        "/mnt/user-data/outputs/score_histograms.png", dpi=300, bbox_inches="tight"
    )
    print("Plot saved to score_histograms.png")

    # Show plot
    plt.show()
