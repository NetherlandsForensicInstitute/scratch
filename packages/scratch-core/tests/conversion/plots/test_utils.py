import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.constants import micro
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from conversion.plots.utils import (
    draw_metadata_box,
    figure_to_array,
    get_bounding_box,
    get_col_widths,
    get_figure_dimensions,
    get_height_ratios,
    get_metadata_dimensions,
    metadata_to_table_data,
    plot_depth_map_on_axes,
    plot_depth_map_with_axes,
    plot_profiles_on_axes,
    plot_side_by_side_on_axes,
)

from .helper_functions import (
    assert_valid_rgb_image,
    create_synthetic_striation_data,
)


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

    def test_wrapping_produces_continuation_rows(self):
        metadata = {"Short": "v", "Long": "A" * 80}
        result = metadata_to_table_data(metadata, wrap_width=25)
        assert result[0][0] == "Short:"
        assert result[1][0] == "Long:"
        # Continuation rows have empty key
        for row in result[2:]:
            assert row[0] == ""

    def test_multiple_keys_with_wrapping(self):
        metadata = {"A": "x" * 60, "B": "y" * 60}
        result = metadata_to_table_data(metadata, wrap_width=25)
        # Total rows should be more than 2 due to wrapping
        assert len(result) > 2


class TestGetColWidths:
    def test_returns_two_widths(self):
        table_data = [["Key:", "Value"]]
        key_w, val_w = get_col_widths(0.06, table_data)
        assert key_w > 0
        assert val_w > 0

    def test_widths_fit_within_margins(self):
        table_data = [["Key:", "Value"]]
        side_margin = 0.10
        key_w, val_w = get_col_widths(side_margin, table_data)
        assert key_w + val_w == pytest.approx(1.0 - 2 * side_margin, abs=0.01)

    def test_long_keys_get_more_width(self):
        short = [["K:", "Value"]]
        long = [["Very Long Key Name:", "V"]]
        key_w_short, _ = get_col_widths(0.06, short)
        key_w_long, _ = get_col_widths(0.06, long)
        assert key_w_long > key_w_short

    def test_key_ratio_clamped_between_35_and_50_percent(self):
        # Very short key compared to long value
        table_data = [["K:", "A" * 100]]
        key_w, val_w = get_col_widths(0.0, table_data)
        total = key_w + val_w
        assert key_w / total >= 0.35 - 0.001


class TestGetBoundingBox:
    def test_returns_bbox(self):
        table_data = [["Key:", "Value"]] * 3
        result = get_bounding_box(0.06, table_data)
        assert isinstance(result, Bbox)

    def test_fewer_rows_give_more_height_per_row(self):
        few = get_bounding_box(0.06, [["K:", "V"]] * 3)
        many = get_bounding_box(0.06, [["K:", "V"]] * 12)
        # Fewer rows -> more generous spacing per row (height/n_rows)
        assert few.height / 3 > many.height / 12

    def test_bbox_centered_vertically(self):
        table_data = [["K:", "V"]] * 5
        bbox = get_bounding_box(0.06, table_data)
        center_y = bbox.y0 + bbox.height / 2
        assert center_y == pytest.approx(0.5, abs=0.05)

    def test_respects_side_margin(self):
        margin = 0.10
        bbox = get_bounding_box(margin, [["K:", "V"]])
        assert bbox.x0 == pytest.approx(margin)
        assert bbox.width == pytest.approx(1.0 - 2 * margin)


class TestGetHeightRatios:
    def test_returns_correct_number_of_ratios(self):
        assert len(get_height_ratios(0.15, 0.30, 0.25)) == 3
        assert len(get_height_ratios(0.15, 0.32, 0.22, 0.20)) == 4

    def test_ratios_sum_to_one(self):
        ratios = get_height_ratios(0.15, 0.40, 0.40)
        assert sum(ratios) == pytest.approx(1.0)

    def test_larger_row0_increases_first_ratio(self):
        small = get_height_ratios(0.10, 0.40, 0.40)
        large = get_height_ratios(0.30, 0.40, 0.40)
        assert large[0] > small[0]

    def test_fixed_rows_decrease_with_larger_row0(self):
        small = get_height_ratios(0.10, 0.40, 0.40)
        large = get_height_ratios(0.30, 0.40, 0.40)
        # Fixed rows should get smaller fraction as row0 grows
        assert large[1] < small[1]


class TestGetMetadataDimensions:
    def test_returns_rows_and_ratio(self):
        meta = {"Key": "Value"}
        rows, ratio = get_metadata_dimensions(meta, meta, wrap_width=25)
        assert rows >= 1
        assert ratio >= 0.12

    def test_more_keys_increase_rows(self):
        short = {"A": "1"}
        long = {"A": "1", "B": "2", "C": "3", "D": "4"}
        rows_short, _ = get_metadata_dimensions(short, short, wrap_width=25)
        rows_long, _ = get_metadata_dimensions(long, long, wrap_width=25)
        assert rows_long > rows_short

    def test_uses_max_of_both_metadata(self):
        short = {"A": "1"}
        long = {"A": "1", "B": "2", "C": "3"}
        rows, _ = get_metadata_dimensions(short, long, wrap_width=25)
        rows_rev, _ = get_metadata_dimensions(long, short, wrap_width=25)
        assert rows == rows_rev

    def test_wrapping_increases_rows(self):
        short_val = {"Key": "Short"}
        long_val = {"Key": "A" * 100}
        rows_short, _ = get_metadata_dimensions(short_val, short_val, wrap_width=25)
        rows_long, _ = get_metadata_dimensions(long_val, long_val, wrap_width=25)
        assert rows_long > rows_short


class TestDrawMetadataBox:
    def test_draws_table_with_border(self):
        fig, ax = plt.subplots()
        metadata = {"Key": "Value", "Other": "Data"}
        draw_metadata_box(ax, metadata, title="Test")
        assert ax.get_title() == "Test"
        plt.close(fig)

    def test_without_border(self):
        fig, ax = plt.subplots()
        draw_metadata_box(ax, {"K": "V"}, draw_border=False)
        for spine in ax.spines.values():
            assert not spine.get_visible()
        plt.close(fig)

    def test_no_title(self):
        fig, ax = plt.subplots()
        draw_metadata_box(ax, {"K": "V"}, title=None)
        assert ax.get_title() == ""
        plt.close(fig)

    def test_empty_key_in_metadata(self):
        fig, ax = plt.subplots()
        metadata = {"Key": "Value", "": "", "Other": "Data"}
        draw_metadata_box(ax, metadata)
        plt.close(fig)


class TestPlotDepthMapWithAxes:
    """Tests for plot_depth_map_with_axes function."""

    def test_returns_rgb_image(self, impression_sample_depth_data: np.ndarray):
        result = plot_depth_map_with_axes(
            data=impression_sample_depth_data,
            scale=1.5 * micro,
            title="Test Surface",
        )
        assert_valid_rgb_image(result)

    def test_handles_nan_values(self):
        data = np.random.randn(50, 60) * micro
        data[10:20, 10:20] = np.nan
        result = plot_depth_map_with_axes(
            data=data, scale=1.5 * micro, title="With NaN"
        )
        assert_valid_rgb_image(result)

    def test_square_data(self):
        data = create_synthetic_striation_data(height=200, width=200, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625 * micro, title="Square")
        assert_valid_rgb_image(result)

    def test_wide_data(self):
        data = create_synthetic_striation_data(height=100, width=400, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625 * micro, title="Wide")
        assert_valid_rgb_image(result)

    def test_tall_data(self):
        data = create_synthetic_striation_data(height=400, width=100, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625 * micro, title="Tall")
        assert_valid_rgb_image(result)

    def test_uniform_data(self):
        data = np.ones((100, 100)) * micro
        result = plot_depth_map_with_axes(data, scale=1.5625 * micro, title="Uniform")
        assert_valid_rgb_image(result)


class TestPlotDepthmapOnAxes:
    def test_creates_image(self, striation_surface_reference):
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(
            ax, fig, striation_surface_reference, 1.5625 * micro, "Test"
        )
        assert len(ax.images) == 1
        plt.close(fig)

    def test_sets_title(self, striation_surface_reference):
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(
            ax, fig, striation_surface_reference, 1.5625 * micro, "My Title"
        )
        assert ax.get_title() == "My Title"
        plt.close(fig)


class TestPlotProfilesOnAxes:
    def test_creates_two_lines(
        self, striation_profile_reference, striation_profile_compared
    ):
        fig, ax = plt.subplots()
        plot_profiles_on_axes(
            ax,
            striation_profile_reference,
            striation_profile_compared,
            1.5625 * micro,
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
            1.5625 * micro,
            0.85,
            "Test",
        )
        assert "Test" in ax.get_title()
        assert "0.85" in ax.get_title()
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)


class TestPlotSideBySideOnAxes:
    def test_creates_combined_image(
        self, striation_surface_reference, striation_surface_compared
    ):
        fig, ax = plt.subplots()
        plot_side_by_side_on_axes(
            ax,
            fig,
            striation_surface_reference,
            striation_surface_compared,
            1.5625 * micro,
        )
        assert len(ax.images) == 1
        plt.close(fig)

    def test_combined_width_includes_gap(
        self, striation_surface_reference, striation_surface_compared
    ):
        fig, ax = plt.subplots()
        plot_side_by_side_on_axes(
            ax,
            fig,
            striation_surface_reference,
            striation_surface_compared,
            1.5625 * micro,
        )
        image_data = ax.images[0].get_array()
        assert image_data is not None
        expected_min_width = (
            striation_surface_reference.shape[1] + striation_surface_compared.shape[1]
        )
        assert image_data.shape[1] > expected_min_width
        plt.close(fig)
