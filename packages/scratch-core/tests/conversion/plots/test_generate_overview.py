"""
This Class is to generate a realistic overview image with synthetic data, so no unit test and will be removed before
merging.
"""

import pytest

from conversion.data_formats import Mark, MarkMetadata
from conversion.profile_correlator import StriationComparisonResults, Profile
from conversion.plots.plot_impression import plot_impression_comparison_results
from conversion.plots.plot_striation import plot_striation_comparison_results
from conversion.surface_comparison.models import (
    ComparisonResult,
    ComparisonParams,
)

from .helper_functions import assert_valid_rgb_image


@pytest.mark.integration
class TestGenerateOverview:
    """Generate the full comparison overview with realistic synthetic data."""

    def test_generates_overview_png(
        self,
        impression_overview_marks: dict[str, Mark],
        impression_overview_cmc_result: ComparisonResult,
        impression_overview_comparison_params: ComparisonParams,
        sample_metadata_reference: MarkMetadata,
        sample_metadata_compared: MarkMetadata,
    ) -> None:
        """Produce plot_results_overview.png and verify it is a valid RGB image."""
        results = plot_impression_comparison_results(
            mark_reference_raw=impression_overview_marks["reference_raw"],
            mark_compared_raw=impression_overview_marks["compared_raw"],
            mark_reference_filtered=impression_overview_marks["reference_filtered"],
            mark_compared_filtered=impression_overview_marks["compared_filtered"],
            cmc_result=impression_overview_cmc_result,
            comparison_params=impression_overview_comparison_params,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        overview = results.comparison_overview
        assert_valid_rgb_image(overview)

    def test_generates_striation_overview_png(
        self,
        striation_mark_reference: Mark,
        striation_mark_compared: Mark,
        striation_mark_reference_aligned: Mark,
        striation_mark_compared_aligned: Mark,
        profile_reference: Profile,
        profile_compared: Profile,
        striation_metrics: StriationComparisonResults,
        sample_metadata_reference: MarkMetadata,
        sample_metadata_compared: MarkMetadata,
    ) -> None:
        """Produce plot_striation_overview.png and verify it is a valid RGB image."""
        results = plot_striation_comparison_results(
            mark_reference=striation_mark_reference,
            mark_compared=striation_mark_compared,
            mark_reference_aligned=striation_mark_reference_aligned,
            mark_compared_aligned=striation_mark_compared_aligned,
            profile_reference_aligned=profile_reference,
            profile_compared_aligned=profile_compared,
            metrics=striation_metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        overview = results.comparison_overview
        assert_valid_rgb_image(overview)
