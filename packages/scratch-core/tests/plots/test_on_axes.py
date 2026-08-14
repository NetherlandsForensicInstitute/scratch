import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.constants import micro

from plots.on_axes import (
    plot_depth_map_with_axes,
    plot_depth_map_on_axes,
    plot_profiles_on_axes,
    plot_side_by_side_on_axes,
)
from .helper_functions import assert_valid_rgb_image, create_synthetic_striation_data


@pytest.mark.integration
class TestPlotDepthMapWithAxes:
    """Tests for plot_depth_map_with_axes function."""

    def test_returns_rgb_image(self, impression_sample_depth_data: np.ndarray):
        result = plot_depth_map_with_axes(
            data=impression_sample_depth_data,
            scale=1.5 * micro,
            title="Test Surface",
        )
        assert_valid_rgb_image(result)

    def test_handles_nan_values(self):
        data = np.random.randn(50, 60) * micro
        data[10:20, 10:20] = np.nan
        result = plot_depth_map_with_axes(
            data=data, scale=1.5 * micro, title="With NaN"
        )
        assert_valid_rgb_image(result)

    def test_square_data(self):
        data = create_synthetic_striation_data(height=200, width=200, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625 * micro, title="Square")
        assert_valid_rgb_image(result)

    def test_wide_data(self):
        data = create_synthetic_striation_data(height=100, width=400, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625 * micro, title="Wide")
        assert_valid_rgb_image(result)

    def test_tall_data(self):
        data = create_synthetic_striation_data(height=400, width=100, seed=42)
        result = plot_depth_map_with_axes(data, scale=1.5625 * micro, title="Tall")
        assert_valid_rgb_image(result)

    def test_uniform_data(self):
        data = np.ones((100, 100)) * micro
        result = plot_depth_map_with_axes(data, scale=1.5625 * micro, title="Uniform")
        assert_valid_rgb_image(result)


class TestPlotDepthmapOnAxes:
    def test_creates_image(self, striation_surface_reference):
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(
            ax, fig, striation_surface_reference, 1.5625 * micro, "Test"
        )
        assert len(ax.images) == 1
        plt.close(fig)

    def test_sets_title(self, striation_surface_reference):
        fig, ax = plt.subplots()
        plot_depth_map_on_axes(
            ax, fig, striation_surface_reference, 1.5625 * micro, "My Title"
        )
        assert ax.get_title() == "My Title"
        plt.close(fig)


class TestPlotProfilesOnAxes:
    def test_creates_two_lines(self, profile_reference, profile_compared):
        fig, ax = plt.subplots()
        plot_profiles_on_axes(
            ax,
            profile_reference.heights,
            profile_compared.heights,
            1.5625 * micro,
            0.85,
            "Test",
        )
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_sets_labels_and_title(self, profile_reference, profile_compared):
        fig, ax = plt.subplots()
        plot_profiles_on_axes(
            ax,
            profile_reference.heights,
            profile_compared.heights,
            1.5625 * micro,
            0.85,
            "Test",
        )
        assert "Test" in ax.get_title()
        assert "0.85" in ax.get_title()
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)


class TestPlotSideBySideOnAxes:
    def test_creates_combined_image(
        self, striation_surface_reference, striation_surface_compared
    ):
        fig, ax = plt.subplots()
        plot_side_by_side_on_axes(
            ax,
            fig,
            striation_surface_reference,
            striation_surface_compared,
            1.5625 * micro,
        )
        assert len(ax.images) == 1
        plt.close(fig)

    def test_combined_width_includes_gap(
        self, striation_surface_reference, striation_surface_compared
    ):
        fig, ax = plt.subplots()
        plot_side_by_side_on_axes(
            ax,
            fig,
            striation_surface_reference,
            striation_surface_compared,
            1.5625 * micro,
        )
        image_data = ax.images[0].get_array()
        assert image_data is not None
        expected_min_width = (
            striation_surface_reference.shape[1] + striation_surface_compared.shape[1]
        )
        assert image_data.shape[1] > expected_min_width
        plt.close(fig)
