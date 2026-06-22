from typing import Sequence

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.constants import micro

from conversion.data_formats import Mark, MarkMetadata
from conversion.plots.data_formats import (
    ImpressionComparisonPlots,
)
from conversion.plots.plot_impression import (
    _plot_cell_heatmap_on_axes,
    plot_cell_overlay_on_axes,
    plot_cell_correlation_heatmap,
    plot_cell_grid_overlay,
    plot_comparison_overview,
    plot_impression_comparison_results,
)
from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)
from conversion.surface_comparison.utils import _cells_correlation_to_grid

from .helper_functions import assert_valid_rgb_image
from ..helper_functions import make_cell


class TestPlotCellGridOverlay:
    """Tests for plot_cell_grid_overlay function."""

    def test_returns_rgb_image(
        self,
        impression_sample_depth_data: np.ndarray,
        impression_overview_cells: Sequence[Cell],
    ):
        result = plot_cell_grid_overlay(
            data=impression_sample_depth_data,
            scale=1.5 * micro,
            cells=impression_overview_cells,
        )
        assert_valid_rgb_image(result)

    def test_show_only_cmc_cells(
        self,
        impression_sample_depth_data: np.ndarray,
        impression_overview_cells: Sequence[Cell],
    ):
        result = plot_cell_grid_overlay(
            data=impression_sample_depth_data,
            scale=1.5 * micro,
            cells=impression_overview_cells,
            show_all_cells=False,
        )
        assert_valid_rgb_image(result)

    def test_custom_threshold(
        self,
        impression_sample_depth_data: np.ndarray,
        impression_overview_cells: Sequence[Cell],
    ):
        result = plot_cell_grid_overlay(
            data=impression_sample_depth_data,
            scale=1.5 * micro,
            cells=impression_overview_cells,
        )
        assert_valid_rgb_image(result)


class TestPlotCellCorrelationHeatmap:
    """Tests for plot_cell_correlation_heatmap function."""

    def test_returns_rgb_image(
        self,
        impression_overview_cells: Sequence[Cell],
    ):
        result = plot_cell_correlation_heatmap(
            cells=impression_overview_cells,
            surface_extent_um=(300.0, 200.0),
        )
        assert_valid_rgb_image(result)

    def test_handles_different_grid_sizes(self):
        rng = np.random.default_rng(42)

        for n_rows, n_cols in [(2, 3), (5, 5), (3, 8)]:
            cell_size = (1e-3 / n_cols, 1e-3 / n_rows)
            cells = [
                make_cell(
                    (
                        c * cell_size[0] + cell_size[0] / 2,
                        r * cell_size[1] + cell_size[1] / 2,
                    ),
                    float(rng.random()),
                    is_congruent=rng.random() > 0.5,
                    cell_size=cell_size,
                )
                for r in range(n_rows)
                for c in range(n_cols)
            ]

            result = plot_cell_correlation_heatmap(
                cells=cells,
                surface_extent_um=(300.0, 200.0),
            )
            assert_valid_rgb_image(result)


@pytest.mark.integration
class TestPlotComparisonOverview:
    """Tests for plot_comparison_overview function."""

    def test_returns_rgb_image(
        self,
        impression_sample_mark: Mark,
        impression_overview_cmc_result: ComparisonResult,
        impression_overview_comparison_params: ComparisonParams,
        sample_metadata_reference: MarkMetadata,
        sample_metadata_compared: MarkMetadata,
    ):
        result = plot_comparison_overview(
            mark_reference_raw=impression_sample_mark,
            mark_compared_raw=impression_sample_mark,
            mark_reference_filtered=impression_sample_mark,
            mark_compared_filtered=impression_sample_mark,
            cmc_result=impression_overview_cmc_result,
            comparison_params=impression_overview_comparison_params,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )
        assert_valid_rgb_image(result)


@pytest.mark.integration
class TestPlotImpressionComparisonResults:
    """Integration tests for the main orchestrator function."""

    def test_all_outputs_are_valid_images(
        self,
        impression_sample_mark: Mark,
        impression_sample_mark_compared: Mark,
        impression_sample_mark_compared_filtered: Mark,
        impression_overview_cmc_result: ComparisonResult,
        impression_overview_comparison_params: ComparisonParams,
        sample_metadata_reference: MarkMetadata,
        sample_metadata_compared: MarkMetadata,
    ):
        result = plot_impression_comparison_results(
            mark_reference_raw=impression_sample_mark,
            mark_compared_raw=impression_sample_mark_compared,
            mark_reference_filtered=impression_sample_mark,
            mark_compared_filtered=impression_sample_mark_compared_filtered,
            cmc_result=impression_overview_cmc_result,
            comparison_params=impression_overview_comparison_params,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        assert isinstance(result, ImpressionComparisonPlots)
        assert_valid_rgb_image(result.comparison_overview)
        assert_valid_rgb_image(result.raw_reference_heatmap)
        assert_valid_rgb_image(result.raw_compared_heatmap)
        assert_valid_rgb_image(result.filtered_reference_heatmap)
        assert_valid_rgb_image(result.filtered_compared_heatmap)
        assert_valid_rgb_image(result.cell_reference_heatmap)
        assert_valid_rgb_image(result.cell_compared_heatmap)
        assert_valid_rgb_image(result.cell_overlay)
        assert_valid_rgb_image(result.cell_cross_correlation)


class TestPlotCellHeatmapOnAxes:
    """Tests for _plot_cell_heatmap_on_axes helper."""

    def test_with_surface_extent(
        self,
        impression_overview_cells: list[Cell],
    ):
        cell_correlations = _cells_correlation_to_grid(impression_overview_cells)

        fig, ax = plt.subplots()
        _plot_cell_heatmap_on_axes(
            ax,
            fig,
            cell_correlations=cell_correlations,
            cells=impression_overview_cells,
            surface_extent_um=(300.0, 200.0),
        )
        assert ax.get_title() == "Cell ACCF Distribution"
        assert "µm" in ax.get_xlabel()
        plt.close(fig)


class TestPlotCellOverlayOnAxes:
    """Tests for plot_cell_overlay_on_axes helper."""

    def test_returns_axes_image(self, impression_sample_depth_data: np.ndarray):
        scale = 1.5 * micro
        cells = [
            make_cell((30e-6, 75e-6), 0.9, is_congruent=True),
            make_cell((90e-6, 75e-6), 0.1),
            make_cell((30e-6, 25e-6), 0.3, is_congruent=True),
            make_cell((90e-6, 25e-6), 0.7, is_congruent=True),
        ]

        fig, ax = plt.subplots()
        im = plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale=scale,
            cells=cells,
        )
        assert im is not None
        assert len(ax.images) == 1
        plt.close(fig)

    def test_custom_label_prefix(self, impression_sample_depth_data: np.ndarray):
        scale = 1.5 * micro
        cells = [
            make_cell((30e-6, 75e-6), 0.5, is_congruent=True),
            make_cell((90e-6, 75e-6), 0.5, is_congruent=True),
        ]

        fig, ax = plt.subplots()
        plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale,
            cells=cells,
            cell_label_prefix="B",
        )
        texts = [t.get_text() for t in ax.texts]
        assert all("B" in t for t in texts)
        plt.close(fig)

    def test_draws_rectangles_and_labels(
        self, impression_sample_depth_data: np.ndarray
    ):
        scale = 1.5 * micro
        cells = [
            make_cell((30e-6, 75e-6), 0.9, is_congruent=True),
            make_cell((90e-6, 75e-6), 0.1),
        ]

        fig, ax = plt.subplots()
        plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale,
            cells=cells,
        )
        assert len(ax.lines) > 0
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_show_only_cmc_draws_subset(self, impression_sample_depth_data: np.ndarray):
        scale = 1.5 * micro
        cells = [
            make_cell(
                (30e-6, 75e-6),
                0.9,
                is_congruent=True,
                angle_deg=10.0,
                center_comparison=(32e-6, 73e-6),
            ),
            make_cell((90e-6, 75e-6), 0.1),
        ]

        fig, ax = plt.subplots()
        plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale,
            cells=cells,
            show_all_cells=False,
        )
        assert len(ax.texts) == 1
        plt.close(fig)

class TestPlotCellOverlaySpace:
    """Tests for the reference vs comparison drawing-space distinction.

    Guards against the bug where _draw_cell_labels always used
    center_comparison and angle_deg, so the reference panel rendered
    comparison-space (rotated, translated) cells instead of the
    axis-aligned reference grid.
    """

    @staticmethod
    def _drawn_rectangles(ax) -> list[np.ndarray]:
        """Each plotted cell rectangle as a (5, 2) array of vertices.

        _draw_cell_labels closes each rectangle (5 points, last == first),
        so we filter the axes' lines to those with exactly 5 vertices.
        """
        rects = []
        for line in ax.get_lines():
            xy = np.column_stack(line.get_data())
            if xy.shape == (5, 2):
                rects.append(xy)
        return rects

    @staticmethod
    def _is_axis_aligned(rect: np.ndarray, tol: float = 1e-6) -> bool:
        """True if every edge of the rectangle is horizontal or vertical."""
        return all(
            abs(dx) <= tol or abs(dy) <= tol for dx, dy in np.diff(rect, axis=0)
        )

    def test_reference_space_is_axis_aligned(
        self, impression_sample_depth_data: np.ndarray
    ):
        """Reference space ignores angle_deg and draws an axis-aligned box."""
        cell = make_cell(
            (1.0e-3, 1.0e-3),
            0.9,
            is_congruent=True,
            angle_deg=90.0,
            center_comparison=(2.0e-3, 0.5e-3),
        )
        fig, ax = plt.subplots()
        plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale=1.5 * micro,
            cells=[cell],
            space="reference",
        )
        rects = self._drawn_rectangles(ax)
        assert len(rects) == 1
        assert self._is_axis_aligned(rects[0]), "reference cell should not be rotated"
        plt.close(fig)

    def test_reference_space_uses_center_reference(
        self, impression_sample_depth_data: np.ndarray
    ):
        """Reference rectangle is centered on center_reference, not center_comparison."""
        cell = make_cell(
            (1.0e-3, 1.0e-3),
            0.9,
            is_congruent=True,
            angle_deg=90.0,
            center_comparison=(2.0e-3, 0.5e-3),
        )
        fig, ax = plt.subplots()
        plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale=1.5 * micro,
            cells=[cell],
            space="reference",
        )
        rect = self._drawn_rectangles(ax)[0]
        cx, cy = rect[:4].mean(axis=0) / 1e6  # centroid back to metres
        assert cx == pytest.approx(cell.center_reference[0])
        assert cy == pytest.approx(cell.center_reference[1])
        plt.close(fig)

    def test_comparison_space_uses_center_comparison(
        self, impression_sample_depth_data: np.ndarray
    ):
        """Comparison space centers the cell on center_comparison."""
        cell = make_cell(
            (1.0e-3, 1.0e-3),
            0.9,
            is_congruent=True,
            angle_deg=90.0,
            center_comparison=(2.0e-3, 0.5e-3),
        )
        fig, ax = plt.subplots()
        plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale=1.5 * micro,
            cells=[cell],
            space="comparison",
        )
        rect = self._drawn_rectangles(ax)[0]
        cx, cy = rect[:4].mean(axis=0) / 1e6
        assert cx == pytest.approx(cell.center_comparison[0])
        assert cy == pytest.approx(cell.center_comparison[1])
        plt.close(fig)

    def test_comparison_space_rotation_is_visible(
        self, impression_sample_depth_data: np.ndarray
    ):
        """A 45deg cell must produce a tilted rectangle in comparison space."""
        cell = make_cell(
            (1.0e-3, 1.0e-3),
            0.9,
            is_congruent=True,
            angle_deg=45.0,
            center_comparison=(1.0e-3, 1.0e-3),
        )
        fig, ax = plt.subplots()
        plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale=1.5 * micro,
            cells=[cell],
            space="comparison",
        )
        rect = self._drawn_rectangles(ax)[0]
        assert not self._is_axis_aligned(rect), "45deg comparison cell should be tilted"
        plt.close(fig)

    def test_default_space_is_comparison(
        self, impression_sample_depth_data: np.ndarray
    ):
        """Omitting space reproduces the original comparison-space behaviour."""
        cell = make_cell(
            (1.0e-3, 1.0e-3),
            0.9,
            is_congruent=True,
            angle_deg=45.0,
            center_comparison=(2.0e-3, 2.0e-3),
        )
        fig, ax = plt.subplots()
        plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale=1.5 * micro,
            cells=[cell],
        )
        rect = self._drawn_rectangles(ax)[0]
        cx, _ = rect[:4].mean(axis=0) / 1e6
        assert cx == pytest.approx(cell.center_comparison[0])
        assert not self._is_axis_aligned(rect)
        plt.close(fig)