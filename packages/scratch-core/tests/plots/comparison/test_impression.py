from typing import Sequence

import numpy as np
import pytest
from scipy.constants import micro

from conversion.data_formats import Mark, MarkMetadata
from plots.comparison.data_formats import ImpressionComparisonPlots
from plots.comparison.impression import (
    plot_cell_grid_overlay,
    plot_cell_correlation_heatmap,
    plot_impression_comparison_results,
    plot_impression_comparison_overview,
)

from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)
from ..helper_functions import assert_valid_rgb_image
from ...helper_functions import make_cell


@pytest.mark.integration
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
            space="comparison",
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
            space="comparison",
        )
        assert_valid_rgb_image(result)


@pytest.mark.integration
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
        result = plot_impression_comparison_overview(
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
