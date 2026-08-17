from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes

from container_models.base import FloatArray1D
from plots.likelihood_ratio.data_formats import (
    DensityData,
    HistogramData,
    LlrTransformationData,
)
from plots.likelihood_ratio.distributions import (
    plot_score_histograms,
    plot_score_llr_transformation,
)
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from ..helper_functions import assert_plot_is_valid_image


def generate_test_data(seed: int = 42):
    """Generate example KM/KNM score data for testing."""
    rng = np.random.default_rng(seed)

    n_knm = 17199
    knm_scores_data = rng.gamma(0.5, 0.5, n_knm)

    n_km_low = 787
    n_km_high = 338
    n_km = n_km_low + n_km_high

    km_scores_data = np.concatenate(
        [
            rng.gamma(1.5, 2, n_km_low),
            rng.uniform(10, 50, n_km_high),
        ]
    )

    scores = np.concatenate([knm_scores_data, km_scores_data])
    labels = np.concatenate([np.zeros(n_knm, dtype=int), np.ones(n_km, dtype=int)])

    return scores, labels


def assert_valid_score_histogram(fig: Figure):
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Score"
    assert ax.get_ylabel() == "Normalized density"

    legend = ax.get_legend()
    if legend:
        assert len(legend.get_texts()) > 0


@pytest.fixture
def densities() -> DensityData:
    x = np.linspace(0, 50, 500)

    scores, labels = generate_test_data()
    knm_scores = scores[labels == 0]
    km_scores = scores[labels == 1]
    kde_knm = gaussian_kde(knm_scores)
    kde_km = gaussian_kde(km_scores)

    return DensityData(
        x=x,
        km_density_at_x=kde_km(x),
        knm_density_at_x=kde_knm(x),
    )


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
def test_plot_score_histograms(
    tmp_path: Path,
    densities: DensityData,
    new_score: float | None,
    bins: int,
    show_density: bool,
) -> None:
    scores, labels = generate_test_data()
    fig, ax = plt.subplots()

    data = HistogramData(
        scores=scores,
        labels=labels,
        bins=bins,
        densities=densities if show_density else None,
        new_score=new_score,
    )
    plot_score_histograms(ax, data)
    assert_plot_is_valid_image(fig, tmp_path)
    assert_valid_score_histogram(fig)
    plt.close()


def verify_plot_properties(
    ax: Axes, expected_num_lines: int, should_have_llr_label: bool
):
    assert len(ax.lines) == expected_num_lines
    assert ax.get_xlabel() == "Score"
    assert ax.get_ylabel() == "LogLR"
    assert ax.get_title() == "LogLR plot (with confidence intervals)"

    # Verify legend entries
    legend = ax.get_legend()
    if legend:
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert "LogLR all" in legend_labels
        assert "LogLR all 5%" in legend_labels
        assert "LogLR all 95%" in legend_labels

        if should_have_llr_label:
            assert "LogLR" in legend_labels
        else:
            assert "LogLR" not in legend_labels


class TestPlotLoglrWithConfidence:
    """Test suite for plot_loglr_with_confidence function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_points = 100
        scores = np.linspace(0, 1, n_points)
        llrs = 5 * (scores - 0.5) ** 2 - 2
        llrs_at5 = llrs - 0.5
        llrs_at95 = llrs + 0.5

        return scores, llrs, llrs_at5, llrs_at95

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "score_llr_point, expected_num_lines, should_have_llr_label",
        [
            ((0.9, -1.2), 5, True),  # With score_llr_point
            (None, 3, False),  # Without score_llr_point
        ],
        ids=["with_score_llr_point", "without_score_llr_point"],
    )
    def test_plot_with_and_without_score_llr_point(
        self,
        tmp_path: Path,
        sample_data: tuple[FloatArray1D, FloatArray1D, FloatArray1D, FloatArray1D],
        score_llr_point: tuple[float, float],
        expected_num_lines: int,
        should_have_llr_label: bool,
    ):
        (scores, llrs, llrs_at5, llrs_at95) = sample_data
        """Test plotting with and without score_llr_point."""
        fig, ax = plt.subplots()

        data = LlrTransformationData(
            scores=scores,
            llrs=llrs,
            llrs_at5=llrs_at5,
            llrs_at95=llrs_at95,
            score_llr_point=score_llr_point,
        )
        plot_score_llr_transformation(ax, data)

        verify_plot_properties(
            ax, expected_num_lines, should_have_llr_label
        )  # Verify that plot was created
        assert_plot_is_valid_image(fig, tmp_path)

        plt.close(fig)
