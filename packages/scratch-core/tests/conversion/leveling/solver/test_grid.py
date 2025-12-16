from conversion.leveling.solver import prepare_2d_grid
from image_generation.data_formats import ScanImage
import numpy as np


def test_grid_matches_scan_image_shape(scan_image_rectangular_with_nans: ScanImage):
    x_grid, y_grid = prepare_2d_grid(scan_image_rectangular_with_nans)
    assert x_grid.shape == (
        scan_image_rectangular_with_nans.width,
        scan_image_rectangular_with_nans.height,
    )
    assert y_grid.shape == (
        scan_image_rectangular_with_nans.width,
        scan_image_rectangular_with_nans.height,
    )


def test_grid_is_a_meshgrid(scan_image_rectangular_with_nans: ScanImage):
    x_grid, y_grid = prepare_2d_grid(scan_image_rectangular_with_nans)

    assert all(
        np.array_equal(x_grid[:, i], x_grid[:, i + 1])
        for i in range(x_grid.shape[1] - 1)
    )
    assert all(
        np.array_equal(y_grid[i, :], y_grid[i + 1, :])
        for i in range(y_grid.shape[0] - 1)
    )
    assert np.all(x_grid[:-1, 0] < x_grid[1:, 0])
    assert np.all(y_grid[0, :-1] < y_grid[0, 1:])


def test_grid_is_centered_around_origin_by_default(
    scan_image_rectangular_with_nans: ScanImage,
):
    x_grid, y_grid = prepare_2d_grid(scan_image_rectangular_with_nans)

    mid_x, mid_y = x_grid.shape[0] // 2, y_grid.shape[1] // 2
    xs, ys = x_grid[:, 0], y_grid[0, :]

    assert xs[mid_x - 1] <= 0.0 <= xs[mid_x]
    assert ys[mid_y - 1] <= 0.0 <= ys[mid_y]


def test_grid_is_translated_by_offset(scan_image_rectangular_with_nans: ScanImage):
    si = scan_image_rectangular_with_nans
    offset_x, offset_y = 0.1, 0.5
    x_grid, y_grid = prepare_2d_grid(si, offset=(offset_x, offset_y))

    xs, ys = x_grid[:, 0], y_grid[0, :]

    assert np.isclose(xs[0], -offset_x)
    assert np.isclose(xs[-1], -offset_x + (si.width - 1) * si.scale_x)
    assert np.isclose(ys[0], -offset_y)
    assert np.isclose(ys[-1], -offset_y + (si.height - 1) * si.scale_y)
