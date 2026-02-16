import numpy as np
import pytest

from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    HistogramData,
    ImpressionComparisonMetrics,
    LlrTransformationData,
)
from conversion.plots.plot_cmc_comparison_overview import plot_cmc_comparison_overview

from .helper_functions import assert_valid_rgb_image


@pytest.fixture
def sample_results_metadata(
    impression_overview_metrics: ImpressionComparisonMetrics,
) -> dict[str, str]:
    metrics = impression_overview_metrics
    n_cell_rows, n_cell_cols = metrics.cell_correlations.shape
    n_cells = n_cell_rows * n_cell_cols
    n_cmc = int(np.sum(metrics.cell_correlations >= metrics.cell_similarity_threshold))
    return {
        "Date report": "2023-02-16",
        "User ID": "test_user",
        "Mark type": "Breech face impression",
        "Collection name": "test_collection",
        "KM model": "Beta-binomial",
        "KNM model": "Binomial",
        "Score type": "CMC",
        "Score (transform)": f"{n_cmc} of {n_cells}",
        "LogLR (5%, 95%)": "4.87 (4.87, 4.87)",
        "LR (5%, 95%)": "7.41e+04 (7.41e+04, 7.41e+04)",
        "# of KM scores": "500",
        "# of KNM scores": "5000",
    }


@pytest.fixture
def sample_histogram_data() -> HistogramData:
    rng = np.random.default_rng(42)
    n_knm, n_km = 5000, 500
    knm_scores = rng.exponential(scale=2.0, size=n_knm)
    km_scores = np.clip(rng.normal(loc=28, scale=5, size=n_km), 0, None)
    scores = np.concatenate([knm_scores, km_scores])
    labels = np.concatenate([np.zeros(n_knm), np.ones(n_km)])
    return HistogramData(
        scores=scores,
        labels=labels,
        bins=20,
        densities=None,
        new_score=None,
    )


@pytest.fixture
def sample_llr_data() -> LlrTransformationData:
    llr_scores = np.linspace(0, 55, 200)
    llrs = np.piecewise(
        llr_scores,
        [llr_scores < 20, llr_scores >= 20],
        [lambda s: -2 + 0.1 * s, lambda s: -2 + 0.35 * (s - 10)],
    )
    return LlrTransformationData(
        scores=llr_scores,
        llrs=llrs,
        llrs_at5=llrs - 0.3,
        llrs_at95=llrs + 0.3,
        score_llr_point=None,
    )


@pytest.mark.integration
class TestPlotCmcComparisonOverview:
    """Tests for plot_cmc_comparison_overview function."""

    def test_returns_valid_rgb_image(
        self,
        impression_overview_marks: dict[str, Mark],
        impression_overview_metrics: ImpressionComparisonMetrics,
        impression_overview_metadata_reference: dict[str, str],
        impression_overview_metadata_compared: dict[str, str],
        sample_results_metadata: dict[str, str],
        sample_histogram_data: HistogramData,
        sample_llr_data: LlrTransformationData,
    ) -> None:
        cmc_score = impression_overview_metrics.cmc_score
        llr_scores = sample_llr_data.scores
        llrs = sample_llr_data.llrs

        histogram_data = HistogramData(
            scores=sample_histogram_data.scores,
            labels=sample_histogram_data.labels,
            bins=sample_histogram_data.bins,
            densities=sample_histogram_data.densities,
            new_score=cmc_score,
        )
        llr_data = LlrTransformationData(
            scores=sample_llr_data.scores,
            llrs=sample_llr_data.llrs,
            llrs_at5=sample_llr_data.llrs_at5,
            llrs_at95=sample_llr_data.llrs_at95,
            score_llr_point=(cmc_score, float(np.interp(cmc_score, llr_scores, llrs))),
        )

        result = plot_cmc_comparison_overview(
            mark_reference_filtered=impression_overview_marks["reference_filtered"],
            mark_compared_filtered=impression_overview_marks["compared_filtered"],
            metrics=impression_overview_metrics,
            metadata_reference=impression_overview_metadata_reference,
            metadata_compared=impression_overview_metadata_compared,
            results_metadata=sample_results_metadata,
            histogram_data=histogram_data,
            llr_data=llr_data,
        )

        assert_valid_rgb_image(result)
        assert result.shape[0] > 500, "Figure height too small"
        assert result.shape[1] > 500, "Figure width too small"
