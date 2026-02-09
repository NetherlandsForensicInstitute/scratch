"""
Test script for score histogram plotting.

This script runs two tests:
1. Test with a new_score (green vertical line)
2. Test without a new_score (no vertical line)
"""

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
    # Generate exact counts for each part
    n_km_low = 787  # 70% of 1125
    n_km_high = 338  # 30% of 1125

    km_scores_data = np.concatenate(
        [np.random.gamma(1.5, 2, n_km_low), np.random.uniform(10, 50, n_km_high)]
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


def test_with_new_score():
    """Test 1: Plot with a new_score (green vertical line)."""
    print("Running Test 1: With new_score...")

    # Generate test data
    scores, y = generate_test_data()

    # Define a new score to highlight
    new_score = 5.0

    # Create plot
    fig, ax, ax2 = plot_score_histograms_kde(
        scores, y, new_score=new_score, bins=50, figsize=(10, 6)
    )

    # Save figure
    plt.savefig(
        "/mnt/user-data/outputs/test1_with_new_score.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"✓ Test 1 complete: Plot saved with green vertical line at score={new_score}"
    )
    plt.close()

    return True


def test_without_new_score():
    """Test 2: Plot without a new_score (no vertical line)."""
    print("\nRunning Test 2: Without new_score...")

    # Generate test data
    scores, y = generate_test_data()

    # Create plot without new_score
    fig, ax, ax2 = plot_score_histograms_kde(
        scores,
        y,
        new_score=None,  # No vertical line
        bins=50,
        figsize=(10, 6),
    )

    # Save figure
    plt.savefig(
        "/mnt/user-data/outputs/test2_without_new_score.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Test 2 complete: Plot saved without vertical line")
    plt.close()

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SCORE HISTOGRAM PLOTTING - TEST SUITE")
    print("=" * 60)

    # Run tests
    test1_passed = test_with_new_score()
    test2_passed = test_without_new_score()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Test 1 (with new_score):    {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Test 2 (without new_score): {'PASSED ✓' if test2_passed else 'FAILED ✗'}")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed successfully!")
        print("\nGenerated files:")
        print("  - test1_with_new_score.png    (with green vertical line)")
        print("  - test2_without_new_score.png (without vertical line)")
    else:
        print("\n❌ Some tests failed. Please check the output.")

    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
