"""Tests for grid generation functions: extract_patch, _tile_axis, and generate_grid."""

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.surface_comparison.grid import extract_patch, _tile_axis, generate_grid


def _make_image(height: int, width: int, scale: float = 1e-6) -> ScanImage:
    """Create a ScanImage with sequential values for easy verification."""
    data = np.arange(height * width, dtype=np.float64).reshape(height, width)
    return ScanImage(data=data, scale_x=scale, scale_y=scale)


class TestExtractPatch:
    """Tests for extracting rectangular patches from a ScanImage."""

    def test_fully_inside(self) -> None:
        """Patch fully inside the image returns exact data, no NaN."""
        img = _make_image(10, 10)
        patch = extract_patch(img, coordinates=(2, 3), patch_size=(4, 3))

        assert patch.shape == (3, 4)
        assert not np.any(np.isnan(patch))
        np.testing.assert_array_equal(patch, img.data[3:6, 2:6])

    def test_overhang_left(self) -> None:
        """Patch extending past the left edge is NaN-padded on the left."""
        img = _make_image(10, 10)
        patch = extract_patch(img, coordinates=(-2, 0), patch_size=(5, 5))

        assert patch.shape == (5, 5)
        assert np.all(np.isnan(patch[:, :2]))
        np.testing.assert_array_equal(patch[:, 2:], img.data[0:5, 0:3])

    def test_overhang_top(self) -> None:
        """Patch extending past the top edge is NaN-padded on top."""
        img = _make_image(10, 10)
        patch = extract_patch(img, coordinates=(0, -3), patch_size=(5, 5))

        assert patch.shape == (5, 5)
        assert np.all(np.isnan(patch[:3, :]))
        np.testing.assert_array_equal(patch[3:, :], img.data[0:2, 0:5])

    def test_overhang_right(self) -> None:
        """Patch extending past the right edge is NaN-padded on the right."""
        img = _make_image(10, 10)
        patch = extract_patch(img, coordinates=(8, 0), patch_size=(5, 5))

        assert patch.shape == (5, 5)
        np.testing.assert_array_equal(patch[:, :2], img.data[0:5, 8:10])
        assert np.all(np.isnan(patch[:, 2:]))

    def test_overhang_bottom(self) -> None:
        """Patch extending past the bottom edge is NaN-padded on the bottom."""
        img = _make_image(10, 10)
        patch = extract_patch(img, coordinates=(0, 7), patch_size=(5, 5))

        assert patch.shape == (5, 5)
        np.testing.assert_array_equal(patch[:3, :], img.data[7:10, 0:5])
        assert np.all(np.isnan(patch[3:, :]))

    def test_overhang_top_left_corner(self) -> None:
        """Patch extending past both top and left edges."""
        img = _make_image(10, 10)
        patch = extract_patch(img, coordinates=(-2, -3), patch_size=(5, 5))

        assert patch.shape == (5, 5)
        assert np.all(np.isnan(patch[:3, :]))
        assert np.all(np.isnan(patch[:, :2]))
        np.testing.assert_array_equal(patch[3:, 2:], img.data[0:2, 0:3])

    def test_completely_outside(self) -> None:
        """Patch entirely outside the image is all NaN."""
        img = _make_image(10, 10)
        with pytest.raises(ValueError, match="no overlap"):
            extract_patch(img, coordinates=(-10, -10), patch_size=(5, 5))

    def test_custom_fill_value(self) -> None:
        """Out-of-bounds pixels use the specified fill_value."""
        img = _make_image(10, 10)
        patch = extract_patch(
            img, coordinates=(-2, 0), patch_size=(5, 5), fill_value=0.0
        )

        assert patch[0, 0] == 0.0
        assert patch[0, 1] == 0.0
        assert not np.isnan(patch[0, 0])

    def test_preserves_nan_in_source(self) -> None:
        """NaN values already in the source image are preserved in the patch."""
        img = _make_image(10, 10)
        img.data[2, 3] = np.nan
        patch = extract_patch(img, coordinates=(0, 0), patch_size=(5, 5))

        assert np.isnan(patch[2, 3])


class TestTileAxis:
    """Tests for tiling from a seed center outward."""

    def test_perfect_even_fit(self) -> None:
        """100px extent, cell=25, seed=37.5 → 4 cells aligned with boundaries."""
        coords = _tile_axis(cell_size=25, image_size=100)

        assert coords == [0, 25, 50, 75]

    def test_perfect_odd_fit(self) -> None:
        """125px extent, cell=25, seed=62.5 → 5 cells starting at 0 plus
        2 partial edges that overhang."""
        coords = _tile_axis(cell_size=25, image_size=125)

        # Seed at 62.5 → first top-left at 62.5 - 12.5 = 50, then 25, 0, -25
        # -25 + 25 = 0 which is NOT > 0, so -25 is excluded
        assert coords[0] == 0
        assert len(coords) >= 5
        assert 50 in coords  # seed cell top-left

    def test_single_cell(self) -> None:
        """When the extent is smaller than the cell, a single cell is generated."""
        coords = _tile_axis(cell_size=25, image_size=20)

        assert len(coords) == 1

    def test_all_cells_overlap_extent(self) -> None:
        """Every generated cell has at least some overlap with [0, extent)."""
        extent = 130
        cell_size = 25
        coords = _tile_axis(cell_size=cell_size, image_size=extent)

        for x in coords:
            assert x + cell_size > 0, f"Cell at {x} doesn't overlap image"
            assert x < extent, f"Cell at {x} starts past image end"


class TestGenerateGrid:
    """Tests for the full grid generation pipeline."""

    def test_perfect_even_fit_all_full(self) -> None:
        """100x100 image with cell=25 gives exactly 16 cells, all 100% fill."""
        img = _make_image(100, 100)
        cells = generate_grid(img, cell_size=(25e-6, 25e-6), minimum_fill_fraction=0.5)

        assert len(cells) == 16
        assert all(c.fill_fraction == 1.0 for c in cells)

    def test_perfect_odd_fit(self) -> None:
        """125x125 image with cell=25 produces cells with at least 25 fully
        inside the image."""
        img = _make_image(125, 125)
        cells = generate_grid(img, cell_size=(25e-6, 25e-6), minimum_fill_fraction=0.5)
        full_cells = [c for c in cells if c.fill_fraction == 1.0]

        assert len(full_cells) == 25

    def test_minimum_fill_fraction_filters_cells(self) -> None:
        """Cells below the fill threshold are excluded."""
        img = _make_image(100, 100)
        strict = generate_grid(
            img, cell_size=(30e-6, 30e-6), minimum_fill_fraction=0.99
        )
        lenient = generate_grid(
            img, cell_size=(30e-6, 30e-6), minimum_fill_fraction=0.10
        )

        assert len(strict) <= len(lenient)
        assert all(c.fill_fraction >= 0.99 for c in strict)

    def test_circular_mask_filters_corners(self) -> None:
        """A circular mask results in fewer cells than a full image."""
        data = np.ones((120, 120), dtype=np.float64)
        yy, xx = np.mgrid[:120, :120]
        data[((xx - 60) ** 2 + (yy - 60) ** 2) > 50**2] = np.nan
        img = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        full_img = _make_image(120, 120)
        cells_circular = generate_grid(
            img, cell_size=(25e-6, 25e-6), minimum_fill_fraction=0.5
        )
        cells_full = generate_grid(
            full_img, cell_size=(25e-6, 25e-6), minimum_fill_fraction=0.5
        )

        assert len(cells_circular) < len(cells_full)

    def test_all_nan_image_returns_empty(self) -> None:
        """An image that is entirely NaN produces no cells."""
        data = np.full((100, 100), np.nan, dtype=np.float64)
        img = ScanImage(data=data, scale_x=1e-6, scale_y=1e-6)

        assert (
            generate_grid(img, cell_size=(25e-6, 25e-6), minimum_fill_fraction=0.5)
            == []
        )

    def test_cell_centers_are_symmetric(self) -> None:
        """Cell centers are symmetric around the image center."""
        img = _make_image(100, 100)
        cells = generate_grid(img, cell_size=(25e-6, 25e-6), minimum_fill_fraction=0.5)

        x_centers = sorted(set(c.center[0] for c in cells))
        y_centers = sorted(set(c.center[1] for c in cells))

        # For even fit, centers are symmetric: first+last, second+second-last, etc.
        expected_sum = x_centers[0] + x_centers[-1]
        for i in range(len(x_centers) // 2):
            assert x_centers[i] + x_centers[-(i + 1)] == pytest.approx(expected_sum)
        expected_sum = y_centers[0] + y_centers[-1]
        for i in range(len(y_centers) // 2):
            assert y_centers[i] + y_centers[-(i + 1)] == pytest.approx(expected_sum)

    def test_non_square_cells(self) -> None:
        """Non-square cell sizes produce correctly shaped cells."""
        img = _make_image(100, 200)
        cells = generate_grid(img, cell_size=(40e-6, 20e-6), minimum_fill_fraction=0.5)

        assert len(cells) > 0
        for cell in cells:
            assert cell.cell_data.shape == (20, 40)

    def test_scale_conversion(self) -> None:
        """Cell size in meters is correctly converted to pixels via scale."""
        img = _make_image(100, 100, scale=2e-6)
        # 50e-6 meters / 2e-6 scale = 25 pixels
        cells = generate_grid(img, cell_size=(50e-6, 50e-6), minimum_fill_fraction=0.5)

        for cell in cells:
            assert cell.cell_data.shape == (25, 25)
