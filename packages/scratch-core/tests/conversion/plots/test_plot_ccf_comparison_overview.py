import numpy as np
import pytest

from conversion.plots.data_formats import (
    DensityDict,
    HistogramData,
    LlrTransformationData,
)
from conversion.plots.plot_ccf_comparison_overview import plot_ccf_comparison_overview

from .helper_functions import (
    assert_valid_rgb_image,
    create_sample_metadata_results,
    create_synthetic_striation_mark,
)


@pytest.fixture
def sample_results_metadata():
    return create_sample_metadata_results()


@pytest.fixture
def sample_histogram_data():
    rng = np.random.default_rng(42)
    knm_scores = rng.beta(2, 5, 1000)
    km_scores = rng.beta(8, 2, 100)
    scores = np.concatenate([knm_scores, km_scores])
    labels = np.concatenate([np.zeros(1000), np.ones(100)])
    return HistogramData(
        scores=scores,
        labels=labels,
        bins=None,
        densities=None,
        new_score=None,
    )


@pytest.fixture
def sample_histogram_data_transformed(sample_histogram_data):
    return HistogramData(
        scores=0.52 + sample_histogram_data.scores * 0.47,
        labels=sample_histogram_data.labels,
        bins=None,
        densities=None,
        new_score=None,
    )


@pytest.fixture
def sample_llr_data(sample_histogram_data_transformed):
    scores_t = sample_histogram_data_transformed.scores
    score_grid = np.linspace(scores_t.min(), scores_t.max(), 100)
    llrs = 5 * (score_grid - 0.75) ** 2 - 2
    return LlrTransformationData(
        scores=score_grid,
        llrs=llrs,
        llrs_at5=llrs - 0.5,
        llrs_at95=llrs + 0.5,
        score_llr_point=None,
    )


class TestPlotCCFComparisonOverview:
    """Test suite for plot_ccf_comparison_overview function."""

    def test_returns_valid_rgb_image(
        self,
        striation_mark_reference,
        striation_mark_compared,
        striation_mark_reference_aligned,
        striation_mark_compared_aligned,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_results_metadata,
        sample_histogram_data,
        sample_histogram_data_transformed,
        sample_llr_data,
    ):
        result = plot_ccf_comparison_overview(
            mark_reference_filtered=striation_mark_reference,
            mark_compared_filtered=striation_mark_compared,
            mark_reference_aligned=striation_mark_reference_aligned,
            mark_compared_aligned=striation_mark_compared_aligned,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            results_metadata=sample_results_metadata,
            histogram_data=sample_histogram_data,
            histogram_data_transformed=sample_histogram_data_transformed,
            llr_data=sample_llr_data,
        )
        assert_valid_rgb_image(result)

    @pytest.mark.parametrize(
        "new_score,score_llr_point",
        [
            (None, None),
            (0.85, None),
            (None, (0.97, 5.19)),
            (0.85, (0.97, 5.19)),
        ],
        ids=["no_markers", "with_new_score", "with_llr_point", "with_both"],
    )
    def test_optional_score_markers(
        self,
        striation_mark_reference,
        striation_mark_compared,
        striation_mark_reference_aligned,
        striation_mark_compared_aligned,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_results_metadata,
        sample_histogram_data,
        sample_histogram_data_transformed,
        sample_llr_data,
        new_score,
        score_llr_point,
    ):
        histogram_data = HistogramData(
            scores=sample_histogram_data.scores,
            labels=sample_histogram_data.labels,
            bins=sample_histogram_data.bins,
            densities=sample_histogram_data.densities,
            new_score=new_score,
        )
        llr_data = LlrTransformationData(
            scores=sample_llr_data.scores,
            llrs=sample_llr_data.llrs,
            llrs_at5=sample_llr_data.llrs_at5,
            llrs_at95=sample_llr_data.llrs_at95,
            score_llr_point=score_llr_point,
        )
        result = plot_ccf_comparison_overview(
            mark_reference_filtered=striation_mark_reference,
            mark_compared_filtered=striation_mark_compared,
            mark_reference_aligned=striation_mark_reference_aligned,
            mark_compared_aligned=striation_mark_compared_aligned,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            results_metadata=sample_results_metadata,
            histogram_data=histogram_data,
            histogram_data_transformed=sample_histogram_data_transformed,
            llr_data=llr_data,
        )
        assert_valid_rgb_image(result)

    def test_with_density_estimates(
        self,
        striation_mark_reference,
        striation_mark_compared,
        striation_mark_reference_aligned,
        striation_mark_compared_aligned,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_results_metadata,
        sample_histogram_data,
        sample_histogram_data_transformed,
        sample_llr_data,
    ):
        x = np.linspace(0, 1, 50)
        densities: DensityDict = {
            "x": x,
            "km_density_at_x": 2 * x,
            "knm_density_at_x": 2 * (1 - x),
        }
        histogram_data = HistogramData(
            scores=sample_histogram_data.scores,
            labels=sample_histogram_data.labels,
            bins=sample_histogram_data.bins,
            densities=densities,
            new_score=None,
        )
        histogram_data_transformed = HistogramData(
            scores=sample_histogram_data_transformed.scores,
            labels=sample_histogram_data_transformed.labels,
            bins=sample_histogram_data_transformed.bins,
            densities=densities,
            new_score=None,
        )
        result = plot_ccf_comparison_overview(
            mark_reference_filtered=striation_mark_reference,
            mark_compared_filtered=striation_mark_compared,
            mark_reference_aligned=striation_mark_reference_aligned,
            mark_compared_aligned=striation_mark_compared_aligned,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            results_metadata=sample_results_metadata,
            histogram_data=histogram_data,
            histogram_data_transformed=histogram_data_transformed,
            llr_data=sample_llr_data,
        )
        assert_valid_rgb_image(result)


@pytest.mark.parametrize(
    "metadata_reference,metadata_compared,suffix",
    [
        (
            {
                "Collection": "firearms_extended_collection_name",
                "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
                "Specimen ID": "bullet_specimen_001_reference",
                "Measurement ID": "striated_mark_measurement_extended",
            },
            {
                "Collection": "firearms_extended_collection_name",
                "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
                "Specimen ID": "bullet_specimen_002_comparison",
                "Measurement ID": "striated_mark_measurement_extended",
            },
            "long_metadata",
        ),
        (
            {"ID": "A1", "Type": "ref"},
            {"ID": "B2", "Type": "comp"},
            "short_metadata",
        ),
    ],
)
class TestMetadataVariants:
    """Test with different metadata lengths and content."""

    def test_handles_various_metadata_lengths(
        self,
        metadata_reference,
        metadata_compared,
        suffix,
        striation_mark_reference,
        striation_mark_compared,
        striation_mark_reference_aligned,
        striation_mark_compared_aligned,
        sample_results_metadata,
        sample_histogram_data,
        sample_histogram_data_transformed,
        sample_llr_data,
    ):
        result = plot_ccf_comparison_overview(
            mark_reference_filtered=striation_mark_reference,
            mark_compared_filtered=striation_mark_compared,
            mark_reference_aligned=striation_mark_reference_aligned,
            mark_compared_aligned=striation_mark_compared_aligned,
            metadata_reference=metadata_reference,
            metadata_compared=metadata_compared,
            results_metadata=sample_results_metadata,
            histogram_data=sample_histogram_data,
            histogram_data_transformed=sample_histogram_data_transformed,
            llr_data=sample_llr_data,
        )
        assert_valid_rgb_image(result)


@pytest.mark.integration
class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_uniform_surface_data(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_results_metadata,
        sample_histogram_data,
        sample_histogram_data_transformed,
        sample_llr_data,
    ):
        uniform_mark = create_synthetic_striation_mark(height=100, width=50, seed=42)
        result = plot_ccf_comparison_overview(
            mark_reference_filtered=uniform_mark,
            mark_compared_filtered=uniform_mark,
            mark_reference_aligned=uniform_mark,
            mark_compared_aligned=uniform_mark,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            results_metadata=sample_results_metadata,
            histogram_data=sample_histogram_data,
            histogram_data_transformed=sample_histogram_data_transformed,
            llr_data=sample_llr_data,
        )
        assert_valid_rgb_image(result)

    def test_surface_with_nan_values(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_results_metadata,
        sample_histogram_data,
        sample_histogram_data_transformed,
        sample_llr_data,
    ):
        mark = create_synthetic_striation_mark(height=100, width=50, seed=42)
        mark.scan_image.data[10:20, 10:20] = np.nan
        result = plot_ccf_comparison_overview(
            mark_reference_filtered=mark,
            mark_compared_filtered=mark,
            mark_reference_aligned=mark,
            mark_compared_aligned=mark,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            results_metadata=sample_results_metadata,
            histogram_data=sample_histogram_data,
            histogram_data_transformed=sample_histogram_data_transformed,
            llr_data=sample_llr_data,
        )
        assert_valid_rgb_image(result)

    def test_different_surface_widths(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_results_metadata,
        sample_histogram_data,
        sample_histogram_data_transformed,
        sample_llr_data,
    ):
        mark_narrow = create_synthetic_striation_mark(height=100, width=25, seed=42)
        mark_wide = create_synthetic_striation_mark(height=100, width=100, seed=43)
        result = plot_ccf_comparison_overview(
            mark_reference_filtered=mark_narrow,
            mark_compared_filtered=mark_wide,
            mark_reference_aligned=mark_narrow,
            mark_compared_aligned=mark_wide,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            results_metadata=sample_results_metadata,
            histogram_data=sample_histogram_data,
            histogram_data_transformed=sample_histogram_data_transformed,
            llr_data=sample_llr_data,
        )
        assert_valid_rgb_image(result)
