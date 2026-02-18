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


@pytest.mark.integration
class TestPlotCmcComparisonOverview:
    """Tests for plot_cmc_comparison_overview function."""

    def test_returns_valid_rgb_image(
        self,
        impression_overview_marks: dict[str, Mark],
        impression_overview_metrics: ImpressionComparisonMetrics,
        impression_overview_metadata_reference: dict[str, str],
        impression_overview_metadata_compared: dict[str, str],
        cmc_results_metadata: dict[str, str],
        cmc_histogram_data: HistogramData,
        cmc_llr_data: LlrTransformationData,
    ) -> None:
        cmc_score = impression_overview_metrics.cmc_score
        llr_scores = cmc_llr_data.scores
        llrs = cmc_llr_data.llrs

        histogram_data = HistogramData(
            scores=cmc_histogram_data.scores,
            labels=cmc_histogram_data.labels,
            bins=cmc_histogram_data.bins,
            densities=cmc_histogram_data.densities,
            new_score=cmc_score,
        )
        llr_data = LlrTransformationData(
            scores=cmc_llr_data.scores,
            llrs=cmc_llr_data.llrs,
            llrs_at5=cmc_llr_data.llrs_at5,
            llrs_at95=cmc_llr_data.llrs_at95,
            score_llr_point=(cmc_score, float(np.interp(cmc_score, llr_scores, llrs))),
        )

        result = plot_cmc_comparison_overview(
            mark_reference_filtered=impression_overview_marks["reference_filtered"],
            mark_compared_filtered=impression_overview_marks["compared_filtered"],
            metrics=impression_overview_metrics,
            metadata_reference=impression_overview_metadata_reference,
            metadata_compared=impression_overview_metadata_compared,
            results_metadata=cmc_results_metadata,
            histogram_data=histogram_data,
            llr_data=llr_data,
        )

        assert_valid_rgb_image(result)
        assert result.shape[0] > 500, "Figure height too small"
        assert result.shape[1] > 500, "Figure width too small"
