"""
This Class is to generate a realistic overview image with synthetic data, so no unit test and will be removed before
merging.
"""

import pytest

from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    StriationComparisonMetrics,
    ImpressionComparisonMetrics,
)
from conversion.plots.plot_impression import plot_impression_comparison_results
from conversion.plots.plot_striation import plot_striation_comparison_results

from .helper_functions import assert_valid_rgb_image


@pytest.mark.integration
class TestGenerateOverview:
    """Generate the full comparison overview with realistic synthetic data."""

    def test_generates_overview_png(
        self,
        impression_overview_marks: dict[str, Mark],
        impression_overview_metrics: ImpressionComparisonMetrics,
        impression_overview_metadata_reference: dict[str, str],
        impression_overview_metadata_compared: dict[str, str],
    ) -> None:
        """Produce plot_results_overview.png and verify it is a valid RGB image."""
        results = plot_impression_comparison_results(
            mark_reference_leveled=impression_overview_marks["reference_leveled"],
            mark_compared_leveled=impression_overview_marks["compared_leveled"],
            mark_reference_filtered=impression_overview_marks["reference_filtered"],
            mark_compared_filtered=impression_overview_marks["compared_filtered"],
            metrics=impression_overview_metrics,
            metadata_reference=impression_overview_metadata_reference,
            metadata_compared=impression_overview_metadata_compared,
        )

        overview = results.comparison_overview
        assert_valid_rgb_image(overview)

    def test_generates_striation_overview_png(
        self,
        striation_mark_reference: Mark,
        striation_mark_compared: Mark,
        striation_mark_reference_aligned: Mark,
        striation_mark_compared_aligned: Mark,
        striation_mark_profile_reference: Mark,
        striation_mark_profile_compared: Mark,
        striation_metrics: StriationComparisonMetrics,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ) -> None:
        """Produce plot_striation_overview.png and verify it is a valid RGB image."""
        results = plot_striation_comparison_results(
            mark_reference=striation_mark_reference,
            mark_compared=striation_mark_compared,
            mark_reference_aligned=striation_mark_reference_aligned,
            mark_compared_aligned=striation_mark_compared_aligned,
            mark_profile_reference_aligned=striation_mark_profile_reference,
            mark_profile_compared_aligned=striation_mark_profile_compared,
            metrics=striation_metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        overview = results.comparison_overview
        assert_valid_rgb_image(overview)
