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
            mark_reference_leveled=impression_sample_mark,
            mark_compared_leveled=impression_sample_mark,
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
            mark_reference_leveled=impression_sample_mark,
            mark_compared_leveled=impression_sample_mark_compared,
            mark_reference_filtered=impression_sample_mark,
            mark_compared_filtered=impression_sample_mark_compared_filtered,
            cmc_result=impression_overview_cmc_result,
            comparison_params=impression_overview_comparison_params,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        assert isinstance(result, ImpressionComparisonPlots)
        assert_valid_rgb_image(result.comparison_overview)
        assert_valid_rgb_image(result.leveled_reference_heatmap)
        assert_valid_rgb_image(result.leveled_compared_heatmap)
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
