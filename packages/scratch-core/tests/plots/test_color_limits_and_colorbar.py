import numpy as np
import pytest
import matplotlib.pyplot as plt

from plots.on_axes import (
    _robust_color_limits,
    plot_depth_map_on_axes,
    _plot_surface_with_colorbar,
)
from plots.cell_overlays import plot_cell_overlay_on_axes
from tests.helper_functions import make_cell


class TestRobustColorLimits:
    """Unit tests for the median +/- k*MAD outlier-robust color scaling."""

    def test_symmetric_around_median(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=5.0, scale=1.0, size=(200, 200))
        vmin, vmax = _robust_color_limits(data, k=3.0)
        median = np.median(data)
        assert (median - vmin) == pytest.approx(vmax - median, rel=1e-9)

    def test_larger_k_gives_wider_range(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(100, 100))
        narrow = _robust_color_limits(data, k=1.0)
        wide = _robust_color_limits(data, k=5.0)
        assert wide[0] < narrow[0]
        assert wide[1] > narrow[1]

    def test_outliers_do_not_blow_out_range(self):
        rng = np.random.default_rng(2)
        data = rng.normal(loc=0.0, scale=1.0, size=(100, 100))
        data_with_outlier = data.copy()
        data_with_outlier[0, 0] = 1000.0

        vmin, vmax = _robust_color_limits(data_with_outlier, k=3.0)
        vmin_clean, vmax_clean = _robust_color_limits(data, k=3.0)

        assert vmax < 100  # nowhere near the 1000.0 outlier
        assert vmax == pytest.approx(vmax_clean, abs=0.5)
        assert vmin == pytest.approx(vmin_clean, abs=0.5)

    def test_ignores_nan_values(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(50, 50))
        data_with_nan = data.copy()
        data_with_nan[:10, :10] = np.nan

        vmin, vmax = _robust_color_limits(data_with_nan, k=3.0)
        vmin_ref, vmax_ref = _robust_color_limits(data, k=3.0)

        assert np.isfinite(vmin)
        assert np.isfinite(vmax)
        assert vmin == pytest.approx(vmin_ref, abs=0.5)
        assert vmax == pytest.approx(vmax_ref, abs=0.5)

    def test_all_nan_returns_default_bounds(self):
        data = np.full((10, 10), np.nan)
        assert _robust_color_limits(data) == (0.0, 1.0)

    def test_constant_data_falls_back_to_unit_range(self):
        data = np.full((10, 10), 3.0)
        vmin, vmax = _robust_color_limits(data)
        assert vmin == pytest.approx(2.0)
        assert vmax == pytest.approx(4.0)

    def test_degenerate_mad_falls_back_to_std(self):
        # >50% identical values -> MAD is 0, must fall back to std instead
        # of collapsing the range to a single point.
        data = np.zeros(100)
        data[:10] = np.linspace(-5, 5, 10)
        vmin, vmax = _robust_color_limits(data, k=2.0)
        assert vmin < 0.0 < vmax
        assert vmax > vmin

    def test_skewed_data_stays_symmetric_about_median(self):
        # Even with a heavy-tailed / skewed distribution, the bound must
        # remain symmetric about the median by construction.
        rng = np.random.default_rng(4)
        data = rng.exponential(scale=1.0, size=(100, 100))
        vmin, vmax = _robust_color_limits(data, k=3.0)
        median = np.median(data)
        assert (median - vmin) == pytest.approx(vmax - median, rel=1e-9)


class TestDepthMapColorbarClipping:
    """
    Tests for the outlier-aware colorbar in plot_depth_map_on_axes: red lines
    at the clip bounds, and true min/max labelled at the extend triangle tips.
    """

    @staticmethod
    def _make_data_with_outlier():
        rng = np.random.default_rng(5)
        data = rng.normal(loc=0.0, scale=1e-6, size=(80, 80))
        data[0, 0] = 1e-3  # far outlier, in meters
        return data

    def test_red_lines_at_clip_bounds(self):
        data = self._make_data_with_outlier()
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        cbar_ax = fig.axes[-1]  # colorbar axes is appended last
        red_lines = [line for line in cbar_ax.lines if line.get_color() == "red"]
        assert len(red_lines) == 2
        plt.close(fig)

    def test_true_min_max_annotated_at_tips(self):
        data = self._make_data_with_outlier()
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        cbar_ax = fig.axes[-1]
        annotation_texts = {t.get_text() for t in cbar_ax.texts}
        true_max_um = float(np.nanmax(data)) * 1e6
        true_min_um = float(np.nanmin(data)) * 1e6
        assert f"{true_max_um:.2f}" in annotation_texts
        assert f"{true_min_um:.2f}" in annotation_texts
        plt.close(fig)

    def test_clip_bounds_tighter_than_true_range_with_outlier(self):
        data = self._make_data_with_outlier()
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        cbar_ax = fig.axes[-1]
        red_line_ys = sorted(
            line.get_ydata()[0] for line in cbar_ax.lines if line.get_color() == "red"
        )
        true_max_um = float(np.nanmax(data)) * 1e6
        assert red_line_ys[1] < true_max_um
        plt.close(fig)

    def test_no_nan_labels_when_data_all_finite(self):
        rng = np.random.default_rng(6)
        data = rng.normal(scale=1e-6, size=(50, 50))
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        cbar_ax = fig.axes[-1]
        for t in cbar_ax.texts:
            assert "nan" not in t.get_text().lower()
        plt.close(fig)


class TestPlotSurfaceWithColorbarClipping:
    """
    Tests for _plot_surface_with_colorbar, which re-clips an already-drawn
    AxesImage to robust bounds and adds the same red-line/tip-label colorbar.
    """

    def test_reclips_image_to_robust_bounds(self):
        rng = np.random.default_rng(7)
        data = rng.normal(scale=1e-6, size=(60, 60))
        data[0, 0] = 1e-3

        fig, ax = plt.subplots()
        im = ax.imshow(data)
        im.set_clim(data.min(), data.max())  # simulate an unclipped image

        _plot_surface_with_colorbar(fig, ax, im, title="Test")

        vmin, vmax = im.get_clim()
        assert vmax < data.max()  # tightened away from the outlier
        plt.close(fig)

    def test_sets_axes_title(self):
        data = np.random.default_rng(8).normal(size=(40, 40))
        fig, ax = plt.subplots()
        im = ax.imshow(data)

        _plot_surface_with_colorbar(fig, ax, im, title="My Surface")

        assert ax.get_title() == "My Surface"
        plt.close(fig)

    def test_true_min_max_reflect_unclipped_data(self):
        rng = np.random.default_rng(9)
        data = rng.normal(scale=1.0, size=(50, 50))
        data[5, 5] = 50.0  # extreme, must still be reported at the tip label
        fig, ax = plt.subplots()
        im = ax.imshow(data)

        _plot_surface_with_colorbar(fig, ax, im, title="Test")

        cbar_ax = fig.axes[-1]
        annotation_texts = {t.get_text() for t in cbar_ax.texts}
        assert f"{data.max():.2f}" in annotation_texts
        plt.close(fig)


class TestCellOverlayColorSigma:
    """plot_cell_overlay_on_axes exposes color_sigma; make sure it's honoured."""

    def test_color_sigma_affects_image_clim(self, impression_sample_depth_data):
        cells = [make_cell((30e-6, 75e-6), 0.9, is_congruent=True)]

        fig1, ax1 = plt.subplots()
        im_tight = plot_cell_overlay_on_axes(
            ax1,
            impression_sample_depth_data,
            scale=1.5e-6,
            cells=cells,
            color_sigma=1.0,
        )
        fig2, ax2 = plt.subplots()
        im_wide = plot_cell_overlay_on_axes(
            ax2,
            impression_sample_depth_data,
            scale=1.5e-6,
            cells=cells,
            color_sigma=5.0,
        )

        vmin_tight, vmax_tight = im_tight.get_clim()
        vmin_wide, vmax_wide = im_wide.get_clim()
        assert vmax_wide > vmax_tight
        assert vmin_wide < vmin_tight
        plt.close(fig1)
        plt.close(fig2)
