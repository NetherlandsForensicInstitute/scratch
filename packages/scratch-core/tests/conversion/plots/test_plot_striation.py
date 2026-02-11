import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from conversion.plots.plot_striation import (
    plot_similarity,
    plot_depth_map_with_axes,
    plot_comparison_overview,
    plot_striation_comparison_results,
    get_wavelength_correlation_plot,
)
from conversion.plots.utils import (
    plot_side_by_side_on_axes,
    plot_depth_map_on_axes,
    plot_profiles_on_axes,
    metadata_to_table_data,
    get_figure_dimensions,
    figure_to_array,
)
from .helper_functions import assert_valid_rgb_image, create_synthetic_striation_data


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


class TestFigureToArray:
    def test_returns_rgb_uint8(self):
        fig = Figure(figsize=(4, 3), dpi=100)
        FigureCanvasAgg(fig)
        arr = figure_to_array(fig)
        assert_valid_rgb_image(arr)

    def test_dimensions_match_figsize_and_dpi(self):
        fig = Figure(figsize=(4, 3), dpi=100)
        FigureCanvasAgg(fig)
        arr = figure_to_array(fig)
        assert arr.shape == (300, 400, 3)


class TestGetFigDimensions:
    @pytest.mark.parametrize(
        "height,width,expected_width",
        [
            (100, 200, 10),
            (200, 100, 10),
            (100, 100, 10),
        ],
    )
    def test_width_is_constant(self, height, width, expected_width):
        fig_height, fig_width = get_figure_dimensions(height, width)
        assert fig_width == expected_width

    def test_height_scales_with_aspect_ratio(self):
        # Wide image -> shorter figure
        h1, _ = get_figure_dimensions(100, 200)
        # Tall image -> taller figure
        h2, _ = get_figure_dimensions(200, 100)
        assert h2 > h1


class TestMetadataToTableData:
    def test_simple_metadata(self):
        metadata = {"Key": "Value"}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert result == [["Key:", "Value"]]

    def test_wrapping_long_values(self):
        metadata = {"Key": "A" * 100}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert len(result) > 1
        assert result[0][0] == "Key:"
        assert result[1][0] == ""  # Continuation has empty key

    def test_empty_value(self):
        metadata = {"Key": ""}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert result == [["Key:", ""]]

    def test_preserves_order(self):
        metadata = {"First": "1", "Second": "2", "Third": "3"}
        result = metadata_to_table_data(metadata, wrap_width=40)
        keys = [row[0] for row in result]
        assert keys == ["First:", "Second:", "Third:"]

    def test_non_string_values_converted(self):
        metadata = {"Number": 42, "Float": 3.14}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert result[0] == ["Number:", "42"]
        assert result[1] == ["Float:", "3.14"]

    def test_empty_key_skips_colon(self):
        metadata = {"Key": "Value", "": ""}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert result[1][0] == ""  # No colon for empty key


class TestPlotProfilesOnAxes:
    def test_creates_two_lines(
        self, striation_profile_reference, striation_profile_compared
    ):
        fig, ax = plt.subplots()
        plot_profiles_on_axes(
            ax,
            striation_profile_reference,
            striation_profile_compared,
            1.5625e-6,
            0.85,
            "Test",
        )
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_sets_labels_and_title(
        self, striation_profile_reference, striation_profile_compared
    ):
        fig, ax = plt.subplots()
        plot_profiles_on_axes(
            ax,
            striation_profile_reference,
            striation_profile_compared,
            1.5625e-6,
            0.85,
            "Test",
        )
        assert "Test" in ax.get_title()
        assert "0.85" in ax.get_title()
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)


class TestPlotDepthmapOnAxes:
    def test_creates_image(self, striation_surface_reference):
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, striation_surface_reference, 1.5625e-6, "Test")
        assert len(ax.images) == 1
        plt.close(fig)

    def test_sets_title(self, striation_surface_reference):
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(
            ax, fig, striation_surface_reference, 1.5625e-6, "My Title"
        )
        assert ax.get_title() == "My Title"
        plt.close(fig)


class TestPlotSideBySideOnAxes:
    def test_creates_combined_image(
        self, striation_surface_reference, striation_surface_compared
    ):
        fig, ax = plt.subplots()
        plot_side_by_side_on_axes(
            ax, fig, striation_surface_reference, striation_surface_compared, 1.5625e-6
        )
        assert len(ax.images) == 1
        plt.close(fig)

    def test_combined_width_includes_gap(
        self, striation_surface_reference, striation_surface_compared
    ):
        fig, ax = plt.subplots()
        plot_side_by_side_on_axes(
            ax, fig, striation_surface_reference, striation_surface_compared, 1.5625e-6
        )
        image_data = ax.images[0].get_array()
        assert image_data is not None
        expected_min_width = (
            striation_surface_reference.shape[1] + striation_surface_compared.shape[1]
        )
        assert image_data.shape[1] > expected_min_width
        plt.close(fig)


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

    def test_plot_depthmap_square_data(self):
        data = create_synthetic_striation_data(height=200, width=200, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625e-6, title="Square")
        assert_valid_rgb_image(result)

    def test_plot_depthmap_wide_data(self):
        data = create_synthetic_striation_data(height=100, width=400, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625e-6, title="Wide")
        assert_valid_rgb_image(result)

    def test_plot_depthmap_tall_data(self):
        data = create_synthetic_striation_data(height=400, width=100, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625e-6, title="Tall")
        assert_valid_rgb_image(result)

    def test_plot_with_nan_values(self):
        data = create_synthetic_striation_data(height=100, width=100, seed=42)
        data[40:60, 40:60] = np.nan
        result = plot_depth_map_with_axes(data, scale=1.5625e-6, title="With NaN")
        assert_valid_rgb_image(result)

    def test_plot_with_uniform_data(self):
        data = np.ones((100, 100)) * 1e-6
        result = plot_depth_map_with_axes(data, scale=1.5625e-6, title="Uniform")
        assert_valid_rgb_image(result)


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
        assert_valid_rgb_image(result.filtered_reference_surface_map)
        assert_valid_rgb_image(result.filtered_compared_surface_map)
        assert_valid_rgb_image(result.side_by_side_surface_map)
        assert_valid_rgb_image(result.wavelength_plot)
