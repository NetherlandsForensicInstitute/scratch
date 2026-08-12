import numpy as np
import pytest
import matplotlib.pyplot as plt

from plots.on_axes import (
    _robust_color_limits,
    plot_depth_map_on_axes,
    _plot_surface_with_colorbar,
)


class TestRobustColorLimits:
    """Unit tests for the median +/- k*(1.4826*MAD) color scaling."""

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

    def test_scale_factor_matches_std_for_normal_data(self):
        """The 1.4826 rescaling means k keeps its "~k sigma" meaning."""
        rng = np.random.default_rng(2)
        data = rng.normal(loc=0.0, scale=2.0, size=(500, 500))
        vmin, vmax = _robust_color_limits(data, k=1.0)
        assert (vmax - vmin) / 2 == pytest.approx(2.0, rel=0.05)

    def test_outliers_do_not_blow_out_range(self):
        rng = np.random.default_rng(3)
        data = rng.normal(loc=0.0, scale=1.0, size=(100, 100))
        with_outlier = data.copy()
        with_outlier[0, 0] = 1000.0

        vmin, vmax = _robust_color_limits(with_outlier, k=3.0)
        vmin_clean, vmax_clean = _robust_color_limits(data, k=3.0)

        assert vmax < 100, "single extreme pixel must not drag the bound up"
        assert vmin == pytest.approx(vmin_clean, abs=0.5)
        assert vmax == pytest.approx(vmax_clean, abs=0.5)

    def test_ignores_nan_values(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(50, 50))
        with_nan = data.copy()
        with_nan[:10, :10] = np.nan

        vmin, vmax = _robust_color_limits(with_nan, k=3.0)
        vmin_ref, vmax_ref = _robust_color_limits(data, k=3.0)

        assert np.isfinite(vmin) and np.isfinite(vmax)
        assert vmin == pytest.approx(vmin_ref, abs=0.5)
        assert vmax == pytest.approx(vmax_ref, abs=0.5)

    def test_all_nan_returns_default_bounds(self):
        assert _robust_color_limits(np.full((10, 10), np.nan)) == (0.0, 1.0)

    def test_constant_data_returns_unit_range_around_value(self):
        vmin, vmax = _robust_color_limits(np.full((10, 10), 3.0))
        assert vmin == pytest.approx(2.0)
        assert vmax == pytest.approx(4.0)

    def test_degenerate_mad_falls_back_to_std(self):
        """>50% identical values gives MAD == 0; must not collapse to a point."""
        data = np.zeros(100)
        data[:10] = np.linspace(-5, 5, 10)
        vmin, vmax = _robust_color_limits(data, k=2.0)
        assert vmin < 0.0 < vmax

    def test_skewed_data_stays_symmetric_about_median(self):
        rng = np.random.default_rng(5)
        data = rng.exponential(scale=1.0, size=(100, 100))
        vmin, vmax = _robust_color_limits(data, k=3.0)
        median = np.median(data)
        assert (median - vmin) == pytest.approx(vmax - median, rel=1e-9)


def _data_with_outlier(seed: int = 6):
    """Normal surface in meters with one far outlier pixel."""
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=1e-6, size=(80, 80))
    data[0, 0] = 1e-3
    return data


class TestDepthMapColorbar:
    """
    The outlier-aware colorbar in plot_depth_map_on_axes: red lines at the
    clip bounds, true min/max labelled at the extend triangle tips.
    """

    @staticmethod
    def _colorbar_axes(fig):
        """The cax appended by make_axes_locatable is added last."""
        return fig.axes[-1]

    @staticmethod
    def _red_line_positions(cbar_ax) -> list[float]:
        return sorted(
            line.get_ydata()[0] for line in cbar_ax.lines if line.get_color() == "red"
        )

    def test_two_red_lines_at_clip_bounds(self):
        data = _data_with_outlier()
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        cbar_ax = self._colorbar_axes(fig)
        positions = self._red_line_positions(cbar_ax)
        expected = _robust_color_limits(data * 1e6, k=3.0)

        assert len(positions) == 2
        assert positions[0] == pytest.approx(expected[0])
        assert positions[1] == pytest.approx(expected[1])
        plt.close(fig)

    def test_clip_bounds_tighter_than_true_range(self):
        data = _data_with_outlier()
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        _, upper = self._red_line_positions(self._colorbar_axes(fig))
        assert upper < float(np.nanmax(data)) * 1e6
        plt.close(fig)

    def test_true_min_max_annotated_at_tips(self):
        data = _data_with_outlier()
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        cbar_ax = self._colorbar_axes(fig)
        annotations = {t.get_text() for t in cbar_ax.texts}
        assert f"{float(np.nanmax(data)) * 1e6:.2f}" in annotations
        assert f"{float(np.nanmin(data)) * 1e6:.2f}" in annotations
        plt.close(fig)

    def test_regular_ticks_do_not_crowd_the_red_lines(self):
        data = _data_with_outlier()
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        cbar_ax = self._colorbar_axes(fig)
        vmin, vmax = self._red_line_positions(cbar_ax)
        margin = 0.06 * (vmax - vmin)
        for tick in cbar_ax.get_yticks():
            assert vmin + margin <= tick <= vmax - margin
        plt.close(fig)

    def test_all_nan_surface_does_not_label_nan(self):
        data = np.full((30, 30), np.nan)
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(ax, fig, data, scale=1.5e-6, title="Test")

        cbar_ax = self._colorbar_axes(fig)
        for text in cbar_ax.texts:
            assert "nan" not in text.get_text().lower()
        plt.close(fig)

    def test_color_sigma_is_honoured(self):
        data = _data_with_outlier()
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(
            ax, fig, data, scale=1.5e-6, title="Test", color_sigma=1.0
        )
        tight = ax.images[0].get_clim()
        plt.close(fig)

        fig, ax = plt.subplots()
        plot_depth_map_on_axes(
            ax, fig, data, scale=1.5e-6, title="Test", color_sigma=5.0
        )
        wide = ax.images[0].get_clim()
        plt.close(fig)

        assert wide[0] < tight[0]
        assert wide[1] > tight[1]


class TestPlotSurfaceWithColorbar:
    """
    _plot_surface_with_colorbar re-clips an already-drawn AxesImage and adds
    the same red-line / tip-label colorbar, reading everything back off `im`.
    """

    def test_reclips_image_away_from_outlier(self):
        rng = np.random.default_rng(7)
        data = rng.normal(scale=1e-6, size=(60, 60))
        data[0, 0] = 1e-3

        fig, ax = plt.subplots()
        im = ax.imshow(data)
        im.set_clim(data.min(), data.max())  # simulate an unclipped image

        _plot_surface_with_colorbar(fig, ax, im, title="Test")

        assert im.get_clim()[1] < data.max()
        plt.close(fig)

    def test_is_idempotent_on_already_clipped_image(self):
        """Safe to call on an image the caller already clipped upstream."""
        rng = np.random.default_rng(8)
        data = rng.normal(scale=1e-6, size=(60, 60))
        data[0, 0] = 1e-3
        vmin, vmax = _robust_color_limits(data, k=3.0)

        fig, ax = plt.subplots()
        im = ax.imshow(data, vmin=vmin, vmax=vmax)
        _plot_surface_with_colorbar(fig, ax, im, title="Test")

        assert im.get_clim() == pytest.approx((vmin, vmax))
        plt.close(fig)

    def test_sets_axes_title(self):
        data = np.random.default_rng(9).normal(size=(40, 40))
        fig, ax = plt.subplots()
        im = ax.imshow(data)
        _plot_surface_with_colorbar(fig, ax, im, title="My Surface")
        assert ax.get_title() == "My Surface"
        plt.close(fig)

    def test_tip_labels_report_unclipped_extremes(self):
        rng = np.random.default_rng(10)
        data = rng.normal(scale=1.0, size=(50, 50))
        data[5, 5] = 50.0

        fig, ax = plt.subplots()
        im = ax.imshow(data)
        _plot_surface_with_colorbar(fig, ax, im, title="Test")

        annotations = {t.get_text() for t in fig.axes[-1].texts}
        assert f"{data.max():.2f}" in annotations
        assert f"{data.min():.2f}" in annotations
        plt.close(fig)

    def test_masked_image_extremes_ignore_masked_pixels(self):
        rng = np.random.default_rng(11)
        data = rng.normal(scale=1.0, size=(50, 50))
        data[:5, :5] = np.nan

        fig, ax = plt.subplots()
        im = ax.imshow(data)
        _plot_surface_with_colorbar(fig, ax, im, title="Test")

        annotations = {t.get_text() for t in fig.axes[-1].texts}
        assert f"{np.nanmax(data):.2f}" in annotations
        plt.close(fig)
