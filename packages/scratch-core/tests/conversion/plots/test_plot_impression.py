import numpy as np
import pytest
from matplotlib import pyplot as plt

from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    ImpressionComparisonMetrics,
    ImpressionComparisonPlots,
)
from conversion.plots.plot_impression import (
    _plot_cell_heatmap_on_axes,
    _plot_cell_overlay_on_axes,
    plot_cell_correlation_heatmap,
    plot_cell_grid_overlay,
    plot_comparison_overview,
    plot_impression_comparison_results,
)

from .helper_functions import assert_valid_rgb_image


class TestPlotCellGridOverlay:
    """Tests for plot_cell_grid_overlay function."""

    def test_returns_rgb_image(
        self,
        impression_sample_depth_data: np.ndarray,
        impression_sample_cell_correlations: np.ndarray,
    ):
        result = plot_cell_grid_overlay(
            data=impression_sample_depth_data,
            scale=1.5e-6,
            cell_correlations=impression_sample_cell_correlations,
        )
        assert_valid_rgb_image(result)

    def test_with_custom_positions_and_rotations(
        self,
        impression_sample_depth_data: np.ndarray,
        impression_sample_cell_correlations: np.ndarray,
    ):
        n_rows, n_cols = impression_sample_cell_correlations.shape
        n_cells = n_rows * n_cols
        scale = 1.5e-6
        h, w = impression_sample_depth_data.shape

        cell_w_um = w * scale * 1e6 / n_cols
        cell_h_um = h * scale * 1e6 / n_rows
        positions = np.zeros((n_cells, 2), dtype=np.float64)
        for i in range(n_rows):
            for j in range(n_cols):
                flat = i * n_cols + j
                positions[flat, 0] = (j + 0.5) * cell_w_um
                positions[flat, 1] = (n_rows - 1 - i + 0.5) * cell_h_um

        rotations = np.deg2rad(np.random.default_rng(0).uniform(-5, 5, n_cells))

        result = plot_cell_grid_overlay(
            data=impression_sample_depth_data,
            scale=scale,
            cell_correlations=impression_sample_cell_correlations,
            cell_positions=positions,
            cell_rotations=rotations,
            cell_size_um=(cell_w_um, cell_h_um),
        )
        assert_valid_rgb_image(result)

    def test_show_only_cmc_cells(
        self,
        impression_sample_depth_data: np.ndarray,
        impression_sample_cell_correlations: np.ndarray,
    ):
        result = plot_cell_grid_overlay(
            data=impression_sample_depth_data,
            scale=1.5e-6,
            cell_correlations=impression_sample_cell_correlations,
            show_all_cells=False,
        )
        assert_valid_rgb_image(result)

    def test_custom_threshold(
        self,
        impression_sample_depth_data: np.ndarray,
        impression_sample_cell_correlations: np.ndarray,
    ):
        result = plot_cell_grid_overlay(
            data=impression_sample_depth_data,
            scale=1.5e-6,
            cell_correlations=impression_sample_cell_correlations,
            cell_similarity_threshold=0.80,
        )
        assert_valid_rgb_image(result)

    def test_with_nan_positions_skips_cells(
        self, impression_sample_depth_data: np.ndarray
    ):
        correlations = np.array([[0.9, 0.1], [0.8, 0.3]])
        n_cells = 4
        positions = np.full((n_cells, 2), np.nan, dtype=np.float64)
        # Only place one CMC cell
        positions[0, :] = [50.0, 100.0]
        rotations = np.zeros(n_cells, dtype=np.float64)

        result = plot_cell_grid_overlay(
            data=impression_sample_depth_data,
            scale=1.5e-6,
            cell_correlations=correlations,
            cell_positions=positions,
            cell_rotations=rotations,
            cell_size_um=(40.0, 40.0),
            show_all_cells=False,
        )
        assert_valid_rgb_image(result)


class TestPlotCellCorrelationHeatmap:
    """Tests for plot_cell_correlation_heatmap function."""

    def test_returns_rgb_image(self, impression_sample_cell_correlations: np.ndarray):
        result = plot_cell_correlation_heatmap(
            cell_correlations=impression_sample_cell_correlations,
            surface_extent_um=(300.0, 200.0),
        )
        assert_valid_rgb_image(result)

    def test_handles_different_grid_sizes(self):
        for rows, cols in [(2, 3), (5, 5), (3, 8)]:
            correlations = np.random.rand(rows, cols)
            result = plot_cell_correlation_heatmap(
                cell_correlations=correlations,
                surface_extent_um=(300.0, 200.0),
            )
            assert_valid_rgb_image(result)

    def test_with_nan_cells(self):
        correlations = np.array([[0.5, np.nan], [0.3, 0.7]])
        result = plot_cell_correlation_heatmap(
            cell_correlations=correlations,
            surface_extent_um=(200.0, 200.0),
        )
        assert_valid_rgb_image(result)


@pytest.mark.integration
class TestPlotComparisonOverview:
    """Tests for plot_comparison_overview function."""

    def test_returns_rgb_image(
        self,
        impression_sample_mark: Mark,
        impression_sample_metrics: ImpressionComparisonMetrics,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        result = plot_comparison_overview(
            mark_reference_leveled=impression_sample_mark,
            mark_compared_leveled=impression_sample_mark,
            mark_reference_filtered=impression_sample_mark,
            mark_compared_filtered=impression_sample_mark,
            metrics=impression_sample_metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )
        assert_valid_rgb_image(result)

    def test_with_custom_cell_positions(
        self,
        impression_sample_mark: Mark,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        cell_correlations = np.array([[0.8, 0.1], [0.3, 0.6]])
        # 4 cells: hardcoded positions for the 3 CMC cells (>= 0.25), NaN for non-CMC
        positions = np.array(
            [[50.0, 120.0], [np.nan, np.nan], [40.0, 30.0], [110.0, 25.0]]
        )
        rotations = np.array([0.08, 0.0, 0.05, -0.03])

        metrics = ImpressionComparisonMetrics(
            area_correlation=0.85,
            cell_correlations=cell_correlations,
            cmc_score=75.0,
            sq_ref=1.5,
            sq_comp=1.6,
            sq_diff=0.4,
            has_area_results=True,
            has_cell_results=True,
            cell_positions_compared=positions,
            cell_rotations_compared=rotations,
            cmc_area_fraction=16.04,
            cutoff_low_pass=5.0,
            cutoff_high_pass=250.0,
            cell_size_um=75.0,
            max_error_cell_position=75.0,
            max_error_cell_angle=6.0,
            cell_similarity_threshold=0.25,
        )
        result = plot_comparison_overview(
            mark_reference_leveled=impression_sample_mark,
            mark_compared_leveled=impression_sample_mark,
            mark_reference_filtered=impression_sample_mark,
            mark_compared_filtered=impression_sample_mark,
            metrics=metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )
        assert_valid_rgb_image(result)

    def test_handles_area_only_metrics(
        self,
        impression_sample_mark: Mark,
        impression_sample_cell_correlations: np.ndarray,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        n_cells = impression_sample_cell_correlations.size
        metrics = ImpressionComparisonMetrics(
            area_correlation=0.85,
            cell_correlations=impression_sample_cell_correlations,
            cmc_score=0.0,
            sq_ref=1.5,
            sq_comp=1.6,
            sq_diff=0.4,
            has_area_results=True,
            has_cell_results=False,
            cell_positions_compared=np.full((n_cells, 2), np.nan),
            cell_rotations_compared=np.full(n_cells, np.nan),
            cmc_area_fraction=0.0,
            cutoff_low_pass=5.0,
            cutoff_high_pass=250.0,
            cell_size_um=125.0,
            max_error_cell_position=75.0,
            max_error_cell_angle=6.0,
        )
        result = plot_comparison_overview(
            mark_reference_leveled=impression_sample_mark,
            mark_compared_leveled=impression_sample_mark,
            mark_reference_filtered=impression_sample_mark,
            mark_compared_filtered=impression_sample_mark,
            metrics=metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )
        assert_valid_rgb_image(result)

    def test_with_all_optional_metrics(
        self,
        impression_sample_mark: Mark,
        impression_sample_cell_correlations: np.ndarray,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        n_cells = impression_sample_cell_correlations.size
        metrics = ImpressionComparisonMetrics(
            area_correlation=0.85,
            cell_correlations=impression_sample_cell_correlations,
            cmc_score=75.0,
            sq_ref=1.5,
            sq_comp=1.6,
            sq_diff=0.4,
            has_area_results=True,
            has_cell_results=True,
            cell_positions_compared=np.full((n_cells, 2), np.nan),
            cell_rotations_compared=np.full(n_cells, np.nan),
            cmc_area_fraction=16.04,
            cutoff_low_pass=5.0,
            cutoff_high_pass=250.0,
            cell_size_um=125.0,
            max_error_cell_position=75.0,
            max_error_cell_angle=6.0,
        )
        result = plot_comparison_overview(
            mark_reference_leveled=impression_sample_mark,
            mark_compared_leveled=impression_sample_mark,
            mark_reference_filtered=impression_sample_mark,
            mark_compared_filtered=impression_sample_mark,
            metrics=metrics,
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
        impression_sample_metrics: ImpressionComparisonMetrics,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        result = plot_impression_comparison_results(
            mark_reference_leveled=impression_sample_mark,
            mark_compared_leveled=impression_sample_mark,
            mark_reference_filtered=impression_sample_mark,
            mark_compared_filtered=impression_sample_mark,
            metrics=impression_sample_metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        assert isinstance(result, ImpressionComparisonPlots)
        assert_valid_rgb_image(result.comparison_overview)
        assert_valid_rgb_image(result.leveled_reference_preview)
        assert_valid_rgb_image(result.leveled_compared_preview)
        assert_valid_rgb_image(result.filtered_reference_preview)
        assert_valid_rgb_image(result.filtered_compared_preview)
        assert_valid_rgb_image(result.cell_reference_preview)
        assert_valid_rgb_image(result.cell_compared_preview)
        assert_valid_rgb_image(result.cell_overlay)
        assert_valid_rgb_image(result.cell_cross_correlation)


class TestPlotCellHeatmapOnAxes:
    """Tests for _plot_cell_heatmap_on_axes helper."""

    def test_with_surface_extent(self, impression_sample_cell_correlations: np.ndarray):
        fig, ax = plt.subplots()
        _plot_cell_heatmap_on_axes(
            ax,
            fig,
            impression_sample_cell_correlations,
            surface_extent_um=(300.0, 200.0),
        )
        assert ax.get_title() == "Cell ACCF Distribution"
        assert "µm" in ax.get_xlabel()
        plt.close(fig)

    def test_custom_threshold_colors(self):
        correlations = np.array([[0.9, 0.1], [0.3, 0.7]])
        fig, ax = plt.subplots()
        _plot_cell_heatmap_on_axes(
            ax,
            fig,
            correlations,
            surface_extent_um=(100.0, 100.0),
            cell_similarity_threshold=0.5,
        )
        plt.close(fig)


class TestPlotCellOverlayOnAxes:
    """Tests for _plot_cell_overlay_on_axes helper."""

    def test_returns_axes_image(self, impression_sample_depth_data: np.ndarray):
        correlations = np.array([[0.9, 0.1], [0.3, 0.7]])
        fig, ax = plt.subplots()
        im = _plot_cell_overlay_on_axes(
            ax, impression_sample_depth_data, 1.5e-6, correlations
        )
        assert im is not None
        assert len(ax.images) == 1
        plt.close(fig)

    def test_custom_label_prefix(self, impression_sample_depth_data: np.ndarray):
        correlations = np.array([[0.5, 0.5]])
        fig, ax = plt.subplots()
        _plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            1.5e-6,
            correlations,
            cell_label_prefix="B",
        )
        # Check that text labels contain "B"
        texts = [t.get_text() for t in ax.texts]
        assert all("B" in t for t in texts)
        plt.close(fig)

    def test_grid_mode_draws_rectangles(self, impression_sample_depth_data: np.ndarray):
        correlations = np.array([[0.9, 0.1], [0.3, 0.7]])
        fig, ax = plt.subplots()
        _plot_cell_overlay_on_axes(
            ax, impression_sample_depth_data, 1.5e-6, correlations
        )
        # Should have lines for cell borders + labels
        assert len(ax.lines) > 0
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_rotated_mode_draws_polygons(
        self, impression_sample_depth_data: np.ndarray
    ):
        correlations = np.array([[0.9, 0.1]])
        h, w = impression_sample_depth_data.shape
        scale = 1.5e-6
        cell_w = w * scale * 1e6 / 2
        cell_h = h * scale * 1e6 / 1
        positions = np.array([[cell_w / 2, cell_h / 2], [float("nan"), float("nan")]])
        rotations = np.array([np.deg2rad(10), 0.0])

        fig, ax = plt.subplots()
        _plot_cell_overlay_on_axes(
            ax,
            impression_sample_depth_data,
            scale,
            correlations,
            cell_positions=positions,
            cell_rotations=rotations,
            cell_size_um=(cell_w, cell_h),
            show_all_cells=False,
        )
        # Only 1 CMC cell placed (the other has NaN position)
        assert len(ax.texts) == 1
        plt.close(fig)
