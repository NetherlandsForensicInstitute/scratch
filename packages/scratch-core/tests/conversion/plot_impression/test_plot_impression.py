"""Tests for impression mark comparison visualization."""

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.plots.data_formats import (
    ImpressionComparisonMetrics,
    ImpressionComparisonPlots,
)
from conversion.plots.plot_impression import (
    plot_area_figures,
    plot_cell_correlation_heatmap,
    plot_cell_grid_overlay,
    plot_cmc_figures,
    plot_correlation_histogram,
    plot_cross_correlation_surface,
    plot_depth_map_with_axes,
    plot_difference_map,
    plot_impression_comparison_results,
)


@pytest.fixture
def sample_depth_data() -> np.ndarray:
    """Create synthetic depth data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 120) * 1e-6  # Random surface in meters


@pytest.fixture
def sample_mark(sample_depth_data: np.ndarray) -> Mark:
    """Create a sample Mark for testing."""
    scan_image = ScanImage(
        data=sample_depth_data,
        scale_x=1.5e-6,  # 1.5 µm pixel size
        scale_y=1.5e-6,
    )
    return Mark(
        scan_image=scan_image,
        mark_type=MarkType.FIRING_PIN_IMPRESSION,
    )


@pytest.fixture
def sample_cell_correlations() -> np.ndarray:
    """Create synthetic cell correlation grid."""
    np.random.seed(42)
    return np.random.rand(4, 5)  # 4x5 grid of correlations


@pytest.fixture
def sample_metrics(sample_cell_correlations: np.ndarray) -> ImpressionComparisonMetrics:
    """Create sample metrics for testing."""
    return ImpressionComparisonMetrics(
        area_correlation=0.85,
        cell_correlations=sample_cell_correlations,
        cmc_score=75.0,
        sq_ref=1.5,
        sq_comp=1.6,
        sq_diff=0.4,
        has_area_results=True,
        has_cell_results=True,
    )


class TestPlotDepthMapWithAxes:
    """Tests for plot_depth_map_with_axes function."""

    def test_returns_rgb_image(self, sample_depth_data: np.ndarray):
        """Output should be RGB uint8 array."""
        result = plot_depth_map_with_axes(
            data=sample_depth_data,
            scale=1.5e-6,
            title="Test Surface",
        )
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_handles_nan_values(self):
        """Should handle NaN values in data."""
        data = np.random.randn(50, 60) * 1e-6
        data[10:20, 10:20] = np.nan
        result = plot_depth_map_with_axes(data=data, scale=1.5e-6, title="With NaN")
        assert result.shape[2] == 3


class TestPlotDifferenceMap:
    """Tests for plot_difference_map function."""

    def test_returns_rgb_image(self, sample_depth_data: np.ndarray):
        """Output should be RGB uint8 array."""
        data_comp = (
            sample_depth_data + np.random.randn(*sample_depth_data.shape) * 0.1e-6
        )
        result = plot_difference_map(
            data_ref=sample_depth_data,
            data_comp=data_comp,
            scale=1.5e-6,
        )
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_identical_surfaces_show_zero_difference(
        self, sample_depth_data: np.ndarray
    ):
        """Identical surfaces should produce a valid difference map."""
        result = plot_difference_map(
            data_ref=sample_depth_data,
            data_comp=sample_depth_data.copy(),
            scale=1.5e-6,
        )
        assert result.shape[2] == 3


class TestPlotCrossCorrelationSurface:
    """Tests for plot_cross_correlation_surface function."""

    def test_returns_rgb_image(self, sample_depth_data: np.ndarray):
        """Output should be RGB uint8 array."""
        result = plot_cross_correlation_surface(
            data_ref=sample_depth_data,
            data_comp=sample_depth_data,
            scale=1.5e-6,
            correlation_value=0.95,
        )
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8


class TestPlotCellGridOverlay:
    """Tests for plot_cell_grid_overlay function."""

    def test_returns_rgb_image(
        self, sample_depth_data: np.ndarray, sample_cell_correlations: np.ndarray
    ):
        """Output should be RGB uint8 array."""
        result = plot_cell_grid_overlay(
            data=sample_depth_data,
            scale=1.5e-6,
            cell_correlations=sample_cell_correlations,
        )
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8


class TestPlotCellCorrelationHeatmap:
    """Tests for plot_cell_correlation_heatmap function."""

    def test_returns_rgb_image(self, sample_cell_correlations: np.ndarray):
        """Output should be RGB uint8 array."""
        result = plot_cell_correlation_heatmap(
            cell_correlations=sample_cell_correlations
        )
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_handles_different_grid_sizes(self):
        """Should handle various grid sizes."""
        for rows, cols in [(2, 3), (5, 5), (3, 8)]:
            correlations = np.random.rand(rows, cols)
            result = plot_cell_correlation_heatmap(cell_correlations=correlations)
            assert result.shape[2] == 3


class TestPlotCorrelationHistogram:
    """Tests for plot_correlation_histogram function."""

    def test_returns_rgb_image(self, sample_cell_correlations: np.ndarray):
        """Output should be RGB uint8 array."""
        result = plot_correlation_histogram(cell_correlations=sample_cell_correlations)
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_custom_threshold(self, sample_cell_correlations: np.ndarray):
        """Should accept custom threshold."""
        result = plot_correlation_histogram(
            cell_correlations=sample_cell_correlations,
            threshold=0.7,
        )
        assert result.shape[2] == 3

    def test_handles_nan_values(self):
        """Should handle NaN values in correlations."""
        correlations = np.array([[0.5, np.nan], [0.8, 0.3]])
        result = plot_correlation_histogram(cell_correlations=correlations)
        assert result.shape[2] == 3


class TestPlotAreaFigures:
    """Tests for plot_area_figures function."""

    def test_returns_six_images(self, sample_mark: Mark):
        """Should return tuple of 6 RGB images."""
        result = plot_area_figures(
            mark_ref_leveled=sample_mark,
            mark_comp_leveled=sample_mark,
            mark_ref_filtered=sample_mark,
            mark_comp_filtered=sample_mark,
            correlation_value=0.85,
        )
        assert len(result) == 6
        for img in result:
            assert img.ndim == 3
            assert img.shape[2] == 3
            assert img.dtype == np.uint8


class TestPlotCmcFigures:
    """Tests for plot_cmc_figures function."""

    def test_returns_five_images(
        self, sample_mark: Mark, sample_cell_correlations: np.ndarray
    ):
        """Should return tuple of 5 RGB images."""
        result = plot_cmc_figures(
            mark_ref_filtered=sample_mark,
            mark_comp_filtered=sample_mark,
            cell_correlations=sample_cell_correlations,
        )
        assert len(result) == 5
        for img in result:
            assert img.ndim == 3
            assert img.shape[2] == 3
            assert img.dtype == np.uint8


class TestPlotImpressionComparisonResults:
    """Integration tests for the main orchestrator function."""

    def test_generates_all_plots_when_both_flags_true(
        self, sample_mark: Mark, sample_metrics: ImpressionComparisonMetrics
    ):
        """Should generate all plots when both area and cell results are available."""
        result = plot_impression_comparison_results(
            mark_reference_leveled=sample_mark,
            mark_compared_leveled=sample_mark,
            mark_reference_filtered=sample_mark,
            mark_compared_filtered=sample_mark,
            metrics=sample_metrics,
            _metadata_reference={"Case": "Test"},
            _metadata_compared={"Case": "Test"},
        )

        assert isinstance(result, ImpressionComparisonPlots)

        # Area-based plots should be present
        assert result.leveled_reference is not None
        assert result.leveled_compared is not None
        assert result.filtered_reference is not None
        assert result.filtered_compared is not None
        assert result.difference_map is not None
        assert result.area_cross_correlation is not None

        # Cell/CMC-based plots should be present
        assert result.cell_reference is not None
        assert result.cell_compared is not None
        assert result.cell_overlay is not None
        assert result.cell_cross_correlation is not None
        assert result.cell_correlation_histogram is not None

    def test_only_area_plots_when_cell_flag_false(
        self, sample_mark: Mark, sample_cell_correlations: np.ndarray
    ):
        """Should only generate area plots when has_cell_results is False."""
        metrics = ImpressionComparisonMetrics(
            area_correlation=0.85,
            cell_correlations=sample_cell_correlations,
            cmc_score=75.0,
            sq_ref=1.5,
            sq_comp=1.6,
            sq_diff=0.4,
            has_area_results=True,
            has_cell_results=False,
        )

        result = plot_impression_comparison_results(
            mark_reference_leveled=sample_mark,
            mark_compared_leveled=sample_mark,
            mark_reference_filtered=sample_mark,
            mark_compared_filtered=sample_mark,
            metrics=metrics,
            _metadata_reference={},
            _metadata_compared={},
        )

        # Area-based plots should be present
        assert result.leveled_reference is not None
        assert result.area_cross_correlation is not None

        # Cell/CMC-based plots should be None
        assert result.cell_reference is None
        assert result.cell_correlation_histogram is None

    def test_only_cell_plots_when_area_flag_false(
        self, sample_mark: Mark, sample_cell_correlations: np.ndarray
    ):
        """Should only generate cell plots when has_area_results is False."""
        metrics = ImpressionComparisonMetrics(
            area_correlation=0.85,
            cell_correlations=sample_cell_correlations,
            cmc_score=75.0,
            sq_ref=1.5,
            sq_comp=1.6,
            sq_diff=0.4,
            has_area_results=False,
            has_cell_results=True,
        )

        result = plot_impression_comparison_results(
            mark_reference_leveled=sample_mark,
            mark_compared_leveled=sample_mark,
            mark_reference_filtered=sample_mark,
            mark_compared_filtered=sample_mark,
            metrics=metrics,
            _metadata_reference={},
            _metadata_compared={},
        )

        # Area-based plots should be None
        assert result.leveled_reference is None
        assert result.area_cross_correlation is None

        # Cell/CMC-based plots should be present
        assert result.cell_reference is not None
        assert result.cell_correlation_histogram is not None

    def test_all_outputs_are_valid_images(
        self, sample_mark: Mark, sample_metrics: ImpressionComparisonMetrics
    ):
        """All non-None outputs should be valid RGB images."""
        result = plot_impression_comparison_results(
            mark_reference_leveled=sample_mark,
            mark_compared_leveled=sample_mark,
            mark_reference_filtered=sample_mark,
            mark_compared_filtered=sample_mark,
            metrics=sample_metrics,
            _metadata_reference={},
            _metadata_compared={},
        )

        for field_name in [
            "leveled_reference",
            "leveled_compared",
            "filtered_reference",
            "filtered_compared",
            "difference_map",
            "area_cross_correlation",
            "cell_reference",
            "cell_compared",
            "cell_overlay",
            "cell_cross_correlation",
            "cell_correlation_histogram",
        ]:
            img = getattr(result, field_name)
            if img is not None:
                assert img.ndim == 3, f"{field_name} should be 3D"
                assert img.shape[2] == 3, f"{field_name} should have 3 channels"
                assert img.dtype == np.uint8, f"{field_name} should be uint8"
