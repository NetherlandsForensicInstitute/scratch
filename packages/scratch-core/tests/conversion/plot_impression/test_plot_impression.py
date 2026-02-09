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
    plot_cell_correlation_heatmap,
    plot_cell_grid_overlay,
    plot_comparison_overview,
    plot_depth_map_with_axes,
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


@pytest.fixture
def sample_metadata_reference() -> dict[str, str]:
    """Create sample metadata for reference mark."""
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1",
        "Specimen ID": "cartridge_1",
    }


@pytest.fixture
def sample_metadata_compared() -> dict[str, str]:
    """Create sample metadata for compared mark."""
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1",
        "Specimen ID": "cartridge_2",
    }


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


class TestPlotComparisonOverview:
    """Tests for plot_comparison_overview function."""

    def test_returns_rgb_image(
        self,
        sample_mark: Mark,
        sample_metrics: ImpressionComparisonMetrics,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        """Should return valid RGB image."""
        result = plot_comparison_overview(
            mark_reference_leveled=sample_mark,
            mark_compared_leveled=sample_mark,
            mark_reference_filtered=sample_mark,
            mark_compared_filtered=sample_mark,
            metrics=sample_metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_handles_area_only_metrics(
        self,
        sample_mark: Mark,
        sample_cell_correlations: np.ndarray,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        """Should work when only area results are available."""
        metrics = ImpressionComparisonMetrics(
            area_correlation=0.85,
            cell_correlations=sample_cell_correlations,
            cmc_score=0.0,
            sq_ref=1.5,
            sq_comp=1.6,
            sq_diff=0.4,
            has_area_results=True,
            has_cell_results=False,
        )
        result = plot_comparison_overview(
            mark_reference_leveled=sample_mark,
            mark_compared_leveled=sample_mark,
            mark_reference_filtered=sample_mark,
            mark_compared_filtered=sample_mark,
            metrics=metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )
        assert result.shape[2] == 3


class TestPlotImpressionComparisonResults:
    """Integration tests for the main orchestrator function."""

    def test_generates_all_plots_when_both_flags_true(
        self,
        sample_mark: Mark,
        sample_metrics: ImpressionComparisonMetrics,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        """Should generate all plots when both area and cell results are available."""
        result = plot_impression_comparison_results(
            mark_reference_leveled=sample_mark,
            mark_compared_leveled=sample_mark,
            mark_reference_filtered=sample_mark,
            mark_compared_filtered=sample_mark,
            metrics=sample_metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        assert isinstance(result, ImpressionComparisonPlots)

        # Comparison overview should always be present
        assert result.comparison_overview is not None

        # Area-based plots should be present
        assert result.leveled_reference is not None
        assert result.leveled_compared is not None
        assert result.filtered_reference is not None
        assert result.filtered_compared is not None

        # Cell/CMC-based plots should be present
        assert result.cell_reference is not None
        assert result.cell_compared is not None
        assert result.cell_overlay is not None
        assert result.cell_cross_correlation is not None

    def test_only_area_plots_when_cell_flag_false(
        self,
        sample_mark: Mark,
        sample_cell_correlations: np.ndarray,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
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
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        # Comparison overview always present
        assert result.comparison_overview is not None

        # Area-based plots should be present
        assert result.leveled_reference is not None

        # Cell/CMC-based plots should be None
        assert result.cell_reference is None
        assert result.cell_cross_correlation is None

    def test_only_cell_plots_when_area_flag_false(
        self,
        sample_mark: Mark,
        sample_cell_correlations: np.ndarray,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
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
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        # Comparison overview always present
        assert result.comparison_overview is not None

        # Area-based plots should be None
        assert result.leveled_reference is None

        # Cell/CMC-based plots should be present
        assert result.cell_reference is not None
        assert result.cell_cross_correlation is not None

    def test_all_outputs_are_valid_images(
        self,
        sample_mark: Mark,
        sample_metrics: ImpressionComparisonMetrics,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ):
        """All non-None outputs should be valid RGB images."""
        result = plot_impression_comparison_results(
            mark_reference_leveled=sample_mark,
            mark_compared_leveled=sample_mark,
            mark_reference_filtered=sample_mark,
            mark_compared_filtered=sample_mark,
            metrics=sample_metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        for field_name in [
            "comparison_overview",
            "leveled_reference",
            "leveled_compared",
            "filtered_reference",
            "filtered_compared",
            "cell_reference",
            "cell_compared",
            "cell_overlay",
            "cell_cross_correlation",
        ]:
            img = getattr(result, field_name)
            if img is not None:
                assert img.ndim == 3, f"{field_name} should be 3D"
                assert img.shape[2] == 3, f"{field_name} should have 3 channels"
                assert img.dtype == np.uint8, f"{field_name} should be uint8"
