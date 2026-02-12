from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytest

from conversion.plots.plot_score_histogram_kde import plot_score_histograms_kde
from ..helper_functions import assert_plot_is_valid_image


def generate_test_data():
    """Generate example data for testing."""
    np.random.default_rng(42)

    n_knm = 171991
    knm_scores_data = np.random.gamma(0.5, 0.5, n_knm)

    n_km = 1125
    n_km_low = 787  # 70% of 1125
    n_km_high = 338  # 30% of 1125

    km_scores_data = np.concatenate(
        [
            np.random.gamma(1.5, 2, n_km_low),
            np.random.uniform(10, 50, n_km_high),
        ]
    )

    scores = np.concatenate([knm_scores_data, km_scores_data])
    labels = np.concatenate([np.zeros(n_knm, dtype=int), np.ones(n_km, dtype=int)])

    return scores, labels


@pytest.mark.integration
@pytest.mark.parametrize(
    "new_score, bins, show_density",
    [
        (5.0, 50, True),
        (3.0, 30, False),
        (None, None, True),
        (None, 50, False),
    ],
)
def test_plot_score_histograms_kde(
    tmp_path: Path, new_score: float | None, bins: int, show_density: bool
) -> None:
    scores, labels = generate_test_data()
    fig, ax = plt.subplots()

    plot_score_histograms_kde(
        scores, labels, new_score=new_score, bins=bins, ax=ax, show_density=show_density
    )
    assert_plot_is_valid_image(fig, tmp_path)
    plt.close()
