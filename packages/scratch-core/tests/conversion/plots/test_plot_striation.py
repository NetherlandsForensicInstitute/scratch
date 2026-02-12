import pytest

from conversion.plots.plot_striation import (
    plot_similarity,
    plot_comparison_overview,
    plot_striation_comparison_results,
    get_wavelength_correlation_plot,
)
from matplotlib import pyplot as plt

from .helper_functions import assert_valid_rgb_image, create_synthetic_striation_data


@pytest.mark.integration
@pytest.mark.parametrize(
    "metadata_reference,metadata_compared,suffix",
    [
        (
            {
                "Collection": "firearms_extended_collection_name",
                "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
                "Specimen ID": "bullet_specimen_001_reference",
                "Measurement ID": "striated_mark_measurement_extended",
                "Additional Info": "Some extra metadata field",
            },
            {
                "Collection": "firearms_extended_collection_name",
                "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
                "Specimen ID": "bullet_specimen_002_comparison",
                "Measurement ID": "striated_mark_measurement_extended",
                "Additional Info": "Another extra field value",
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
def test_plot_comparison_overview_metadata_variants(
    striation_mark_reference,
    striation_mark_compared,
    striation_mark_reference_aligned,
    striation_mark_compared_aligned,
    striation_mark_profile_reference,
    striation_mark_profile_compared,
    striation_metrics,
    metadata_reference,
    metadata_compared,
    suffix,
):
    result = plot_comparison_overview(
        mark_reference=striation_mark_reference,
        mark_compared=striation_mark_compared,
        mark_reference_aligned=striation_mark_reference_aligned,
        mark_compared_aligned=striation_mark_compared_aligned,
        mark_profile_reference=striation_mark_profile_reference,
        mark_profile_compared=striation_mark_profile_compared,
        metrics=striation_metrics,
        metadata_reference=metadata_reference,
        metadata_compared=metadata_compared,
    )
    assert_valid_rgb_image(result)


class TestGetWavelengthCorrelationPlot:
    def test_creates_line_plot(self, striation_quality_passbands):
        fig, ax = plt.subplots()
        get_wavelength_correlation_plot(ax, striation_quality_passbands)
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_y_axis_scaled_to_percentage(self, striation_quality_passbands):
        fig, ax = plt.subplots()
        get_wavelength_correlation_plot(ax, striation_quality_passbands)
        ymin, ymax = ax.get_ylim()
        assert ymin == -0.05
        assert ymax == 1.05
        plt.close(fig)

    def test_x_ticks_match_passbands(self, striation_quality_passbands):
        fig, ax = plt.subplots()
        get_wavelength_correlation_plot(ax, striation_quality_passbands)
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(tick_labels) == len(striation_quality_passbands)
        plt.close(fig)

    def test_empty_passbands(self):
        fig, ax = plt.subplots()
        # Should handle gracefully or raise clear error
        with pytest.raises((ValueError, IndexError)):
            get_wavelength_correlation_plot(ax, {})
        plt.close(fig)


@pytest.mark.integration
class TestEdgeCases:
    def test_plot_similarity_identical_profiles(self):
        profile = create_synthetic_striation_data(height=1, width=200, seed=42)
        result = plot_similarity(profile, profile, scale=1.5625e-6, score=1.0)
        assert_valid_rgb_image(result)

    def test_plot_similarity_different_lengths(self):
        profile_short = create_synthetic_striation_data(height=1, width=100, seed=42)
        profile_long = create_synthetic_striation_data(height=1, width=200, seed=43)
        result = plot_similarity(
            profile_short, profile_long, scale=1.5625e-6, score=0.5
        )
        assert_valid_rgb_image(result)


@pytest.mark.integration
class TestStriationComparisonPlotsIntegration:
    def test_all_outputs_are_valid_images(
        self,
        striation_mark_reference,
        striation_mark_compared,
        striation_mark_reference_aligned,
        striation_mark_compared_aligned,
        striation_mark_profile_reference,
        striation_mark_profile_compared,
        striation_metrics,
        sample_metadata_reference,
        sample_metadata_compared,
    ):
        result = plot_striation_comparison_results(
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

        assert_valid_rgb_image(result.similarity_plot)
        assert_valid_rgb_image(result.comparison_overview)
        assert_valid_rgb_image(result.filtered_reference_heatmap)
        assert_valid_rgb_image(result.filtered_compared_heatmap)
        assert_valid_rgb_image(result.side_by_side_heatmap)
        assert_valid_rgb_image(result.wavelength_plot)
