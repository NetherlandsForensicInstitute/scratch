import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from plots.utils import (
    figure_to_array,
    get_figure_dimensions,
    get_height_ratios,
)

from .helper_functions import (
    assert_valid_rgb_image,
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
