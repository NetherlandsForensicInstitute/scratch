import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from plots.utils import (
    figure_to_array,
    get_figure_dimensions,
    get_height_ratios,
    _fit_cell_label_fontsizes,
    side_by_side_gap_width,
    overview_figure_height,
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


class TestSideBySideGapWidth:
    @pytest.mark.parametrize(
        "width_ref,width_comp,expected",
        [
            (200, 200, 2),
            (250, 400, 3),  # ceil(2.5) from the narrower surface
            (400, 250, 3),  # same, arguments swapped
            (100, 100, 1),
            (1, 500, 1),  # never zero for non-empty data
        ],
    )
    def test_gap_from_narrower_surface(self, width_ref, width_comp, expected):
        ref = np.zeros((10, width_ref))
        comp = np.zeros((10, width_comp))
        assert side_by_side_gap_width(ref, comp) == expected

    def test_symmetric_in_arguments(self):
        ref = np.zeros((10, 137))
        comp = np.zeros((10, 462))
        assert side_by_side_gap_width(ref, comp) == side_by_side_gap_width(comp, ref)

    def test_gap_is_narrow_relative_to_surfaces(self):
        """The gap is a visual separator, not a panel; keep it ~1% of width."""
        ref = np.zeros((10, 500))
        comp = np.zeros((10, 500))
        assert side_by_side_gap_width(ref, comp) <= 500 * 0.02


class TestOverviewFigureHeight:
    def test_grows_with_metadata_rows(self):
        assert overview_figure_height(20, 12, 10.0, 20.0) > overview_figure_height(
            5, 12, 10.0, 20.0
        )

    def test_clamped_to_maximum(self):
        assert overview_figure_height(500, 12, 10.0, 15.0) == 15.0

    def test_clamped_to_minimum(self):
        assert overview_figure_height(0, 8, 10.0, 15.0) == 10.0

    def test_linear_between_bounds(self):
        assert overview_figure_height(10, 12, 10.0, 20.0) == pytest.approx(13.2)

    def test_never_leaves_the_clamp_range(self):
        for rows in range(0, 200, 7):
            height = overview_figure_height(rows, 12, 10.0, 15.0)
            assert 10.0 <= height <= 15.0


class TestFitCellLabelFontsizes:
    """
    Labels are shrunk to fit their cell. The helper forces a draw first
    because with aspect="equal" transData is not final until draw time.
    """

    @staticmethod
    def _axes_with_label(text: str = "A12", fontsize: float = 11.0):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        label = ax.text(50, 50, text, fontsize=fontsize, ha="center", va="center")
        return fig, ax, label

    def test_narrow_cell_shrinks_label(self):
        fig, ax, label = self._axes_with_label()
        _fit_cell_label_fontsizes(ax, [label], cell_w_um=2.0)
        assert label.get_fontsize() < 11.0
        plt.close(fig)

    def test_respects_min_fontsize_clamp(self):
        fig, ax, label = self._axes_with_label()
        _fit_cell_label_fontsizes(ax, [label], cell_w_um=0.01, min_fontsize=3.0)
        assert label.get_fontsize() == pytest.approx(3.0)
        plt.close(fig)

    def test_respects_max_fontsize_clamp(self):
        fig, ax, label = self._axes_with_label()
        _fit_cell_label_fontsizes(ax, [label], cell_w_um=100.0, max_fontsize=11.0)
        assert label.get_fontsize() == pytest.approx(11.0)
        plt.close(fig)

    def test_wider_cell_allows_larger_font(self):
        fig, ax, narrow = self._axes_with_label()
        _fit_cell_label_fontsizes(ax, [narrow], cell_w_um=5.0)
        narrow_size = narrow.get_fontsize()
        plt.close(fig)

        fig, ax, wide = self._axes_with_label()
        _fit_cell_label_fontsizes(ax, [wide], cell_w_um=20.0)
        wide_size = wide.get_fontsize()
        plt.close(fig)

        assert wide_size > narrow_size

    def test_longer_label_gets_smaller_font(self):
        fig, ax, short = self._axes_with_label("A1")
        _fit_cell_label_fontsizes(ax, [short], cell_w_um=10.0)
        short_size = short.get_fontsize()
        plt.close(fig)

        fig, ax, long = self._axes_with_label("A1234567")
        _fit_cell_label_fontsizes(ax, [long], cell_w_um=10.0)
        long_size = long.get_fontsize()
        plt.close(fig)

        assert long_size < short_size

    def test_empty_label_is_left_alone(self):
        fig, ax, label = self._axes_with_label(text="", fontsize=7.0)
        _fit_cell_label_fontsizes(ax, [label], cell_w_um=1.0)
        assert label.get_fontsize() == pytest.approx(7.0)
        plt.close(fig)

    def test_scales_each_label_independently(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        short = ax.text(25, 50, "A1", fontsize=11, ha="center", va="center")
        long = ax.text(75, 50, "A123456", fontsize=11, ha="center", va="center")

        _fit_cell_label_fontsizes(ax, [short, long], cell_w_um=10.0)

        assert short.get_fontsize() != long.get_fontsize()
        plt.close(fig)

    def test_empty_text_list_is_a_noop(self):
        fig, ax = plt.subplots()
        _fit_cell_label_fontsizes(ax, [], cell_w_um=10.0)
        plt.close(fig)
