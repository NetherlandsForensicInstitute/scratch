import numpy as np
import matplotlib.pyplot as plt
from conversion.plots.plot_score_histogram_kde import plot_score_histograms_kde


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


def test_default_plt_with_new_score():
    """
    Test 1: Using default plt (creates new figure) with new_score.

    This test demonstrates:
    - Using ax=plt (default parameter)
    - Adding a green vertical line at new_score=5.0
    - Including KDE density curves (show_density=True, default)
    """
    print("Running Test 1: Default plt with new_score...")

    # Generate test data
    scores, y = generate_test_data()

    # Define a new score to highlight
    new_score = 5.0

    # Create figure and plot using default plt
    plt.figure(figsize=(10, 6))
    plot_score_histograms_kde(scores, y, new_score=new_score, bins=50)
    plt.title("Test 1: Score histograms with KDE (default plt)")

    # Save figure
    plt.savefig(
        "/mnt/user-data/outputs/test_1_default_plt.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"✓ Test 1 complete: Plot saved with green vertical line at score={new_score}"
    )
    return True


def test_custom_axis_with_new_score():
    """
    Test 2: Using custom axis with new_score.

    This test demonstrates:
    - Using a custom matplotlib Axes object
    - Adding a green vertical line at new_score=7.5
    - Including KDE density curves
    - Customizing the axis after plotting
    """
    print("\nRunning Test 2: Custom axis with new_score...")

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

    # Save figure
    plt.savefig(
        "/mnt/user-data/outputs/test_2_custom_axis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"✓ Test 2 complete: Plot saved with custom axis and green line at score={new_score}"
    )
    return True


def test_without_density_curves():
    """
    Test 3: Without density curves and without new_score.

    This test demonstrates:
    - Using a custom matplotlib Axes object
    - Disabling KDE density curves (show_density=False)
    - No green vertical line (new_score=None)
    - Only showing histograms
    """
    print("\nRunning Test 3: Without density curves...")

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

    # Save figure
    plt.savefig(
        "/mnt/user-data/outputs/test_3_no_density.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("✓ Test 3 complete: Plot saved without KDE curves or vertical line")
    return True


def test_bonus_subplot_integration():
    """
    Bonus Test: Multiple plots in subplots.

    This test demonstrates:
    - Using the function in a subplot layout
    - Different configurations in each subplot
    - Integration with matplotlib's subplot system
    """
    print("\nRunning Bonus Test: Subplot integration...")

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

    # Save figure
    plt.savefig(
        "/mnt/user-data/outputs/test_bonus_subplots.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("✓ Bonus test complete: 2x2 subplot layout saved")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("PLOT_SCORE_HISTOGRAMS_KDE - TEST SUITE")
    print("=" * 70)

    # Run tests
    test1_passed = test_default_plt_with_new_score()
    test2_passed = test_custom_axis_with_new_score()
    test3_passed = test_without_density_curves()
    bonus_passed = test_bonus_subplot_integration()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Test 1 (default plt):       {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Test 2 (custom axis):       {'PASSED ✓' if test2_passed else 'FAILED ✗'}")
    print(f"Test 3 (no density):        {'PASSED ✓' if test3_passed else 'FAILED ✗'}")
    print(f"Bonus Test (subplots):      {'PASSED ✓' if bonus_passed else 'FAILED ✗'}")
    print("=" * 70)

    all_passed = test1_passed and test2_passed and test3_passed and bonus_passed

    if all_passed:
        print("\n🎉 All tests passed successfully!")
        print("\nGenerated files:")
        print("  - test_1_default_plt.png      (default plt with new_score)")
        print("  - test_2_custom_axis.png      (custom axis with new_score)")
        print("  - test_3_no_density.png       (histograms only)")
        print("  - test_bonus_subplots.png     (2x2 subplot layout)")
    else:
        print("\n❌ Some tests failed. Please check the output.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
