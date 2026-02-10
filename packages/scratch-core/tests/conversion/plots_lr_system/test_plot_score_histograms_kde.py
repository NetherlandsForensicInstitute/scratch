from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from conversion.plots.plot_score_histogram_kde import plot_score_histograms_kde
from PIL import Image
from matplotlib.figure import Figure


def generate_test_data():
    """Generate example data for testing."""
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
    y = np.concatenate([np.zeros(n_knm, dtype=int), np.ones(n_km, dtype=int)])

    print(
        f"  Generated {len(knm_scores_data)} KNM scores and {len(km_scores_data)} KM scores"
    )
    print(f"  Total: {len(scores)} scores, {len(y)} labels")
    assert len(scores) == len(y), "Score and label arrays must have same length!"

    return scores, y


def assert_plot_is_valid_image(fig: Figure, tmp_path: Path) -> None:
    # Create file path inside pytest temp dir
    img_path = tmp_path / "plot.png"

    # Save figure
    fig.savefig(img_path, format="png")

    assert img_path.exists()
    assert img_path.stat().st_size > 0

    # Validate it's a real image
    img = Image.open(img_path)
    img.verify()


def test_default_plt_with_new_score(tmp_path: Path) -> None:
    """
    Test 1: Using default plt (creates new figure) with new_score.

    This test demonstrates:
    - Using ax=plt (default parameter)
    - Adding a green vertical line at new_score=5.0
    - Including KDE density curves (show_density=True, default)
    """

    # Generate test data
    scores, y = generate_test_data()

    # Define a new score to highlight
    new_score = 5.0

    # Create figure and plot using default settings
    fig, ax = plt.subplots()

    # Plot using custom axis
    plot_score_histograms_kde(scores, y, new_score=new_score, bins=50, ax=ax)
    plt.title("Test 1: Score histograms with KDE (default plt)")
    assert_plot_is_valid_image(fig, tmp_path)
    plt.close()


def test_custom_axis_with_new_score(tmp_path: Path) -> None:
    """
    Test 2: Using custom axis with new_score.

    This test demonstrates:
    - Using a custom matplotlib Axes object
    - Adding a green vertical line at new_score=7.5
    - Including KDE density curves
    - Customizing the axis after plotting
    """

    # Generate test data
    scores, y = generate_test_data()

    # Define a new score to highlight
    new_score = 7.5

    # Create figure with custom axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot using custom axis
    plot_score_histograms_kde(scores, y, new_score=new_score, bins=50, ax=ax)

    # Customize the axis after plotting
    ax.set_title("Test 2: Score histograms with KDE (custom axis)")

    assert_plot_is_valid_image(fig, tmp_path)
    plt.close()


def test_without_density_curves(tmp_path: Path) -> None:
    """
    Test 3: Without density curves and without new_score.

    This test demonstrates:
    - Using a custom matplotlib Axes object
    - Disabling KDE density curves (show_density=False)
    - No green vertical line (new_score=None)
    - Only showing histograms
    """

    # Generate test data
    scores, y = generate_test_data()

    # Create figure with custom axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot without density curves and without new_score
    plot_score_histograms_kde(
        scores,
        y,
        new_score=None,  # No vertical line
        bins=50,
        ax=ax,
        show_density=False,  # No KDE curves
    )

    # Customize the axis
    ax.set_title("Test 3: Score histograms only (no KDE)")

    assert_plot_is_valid_image(fig, tmp_path)
    plt.close()


def test_bonus_subplot_integration(tmp_path: Path) -> None:
    """
    Bonus Test: Multiple plots in subplots.

    This test demonstrates:
    - Using the function in a subplot layout
    - Different configurations in each subplot
    - Integration with matplotlib's subplot system
    """

    # Generate test data
    scores, y = generate_test_data()

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Default with new_score
    plot_score_histograms_kde(scores, y, new_score=5.0, bins=50, ax=axes[0, 0])
    axes[0, 0].set_title("With KDE and new_score=5.0")

    # Plot 2: Without new_score
    plot_score_histograms_kde(scores, y, new_score=None, bins=50, ax=axes[0, 1])
    axes[0, 1].set_title("With KDE, no new_score")

    # Plot 3: Without KDE curves
    plot_score_histograms_kde(
        scores, y, new_score=3.0, bins=50, ax=axes[1, 0], show_density=False
    )
    axes[1, 0].set_title("Without KDE, new_score=3.0")

    # Plot 4: Minimal (no KDE, no new_score)
    plot_score_histograms_kde(
        scores, y, new_score=None, bins=30, ax=axes[1, 1], show_density=False
    )
    axes[1, 1].set_title("Minimal: histograms only (30 bins)")

    plt.tight_layout()

    assert_plot_is_valid_image(fig, tmp_path)
    plt.close()
