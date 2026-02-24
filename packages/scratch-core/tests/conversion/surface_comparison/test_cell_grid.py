import numpy as np

from conversion.surface_comparison.grid import _find_grid_origin, generate_grid_centers
from conversion.surface_comparison.models import ComparisonParams, SurfaceMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surface_map(
    height_map: np.ndarray, pixel_spacing_um: float = 1.0
) -> SurfaceMap:
    spacing = np.array([pixel_spacing_um, pixel_spacing_um])
    rows, cols = height_map.shape
    center = np.array([cols * spacing[0] / 2.0, rows * spacing[1] / 2.0])
    return SurfaceMap(
        height_map=height_map, pixel_spacing=spacing, global_center=center
    )


# ---------------------------------------------------------------------------
# _find_grid_origin
# ---------------------------------------------------------------------------


def test_find_optimal_cell_origin_returns_physical_coordinate():
    """_find_grid_origin returns a 2-element physical coordinate in µm.

    For a fully valid 100×100 image with a 20×20 µm cell at 1 µm/px the
    origin must lie within the image bounds [0, 100) in both axes.
    """
    surface = _make_surface_map(np.ones((100, 100)), pixel_spacing_um=1.0)
    params = ComparisonParams(cell_size=np.array([20.0, 20.0]))

    origin = _find_grid_origin(surface, params)

    assert origin.shape == (2,)
    assert 0.0 <= origin[0] < 100.0
    assert 0.0 <= origin[1] < 100.0


# ---------------------------------------------------------------------------
# generate_grid_centers  (not covered elsewhere)
# ---------------------------------------------------------------------------


def test_generate_cell_centers_spacing():
    """Adjacent centers are separated by exactly cell_size in both axes."""
    surface = _make_surface_map(np.ones((100, 100)), pixel_spacing_um=1.0)
    params = ComparisonParams(cell_size=np.array([25.0, 25.0]))

    origin = _find_grid_origin(surface, params)
    centers = generate_grid_centers(surface, origin, params)

    assert len(centers) > 0
    x_unique = np.unique(centers[:, 0])
    y_unique = np.unique(centers[:, 1])
    assert np.allclose(np.diff(x_unique), 25.0)
    assert np.allclose(np.diff(y_unique), 25.0)


def test_generate_cell_centers_all_overlap_image():
    """Every returned cell at least partially overlaps the reference image.

    _axis_centers extends the grid until cells fall *entirely* outside the
    image, so cells whose edge just reaches the image boundary are included.
    The correct check is therefore that each cell's footprint overlaps the
    image, not that the center itself lies within it.
    """
    surface = _make_surface_map(np.ones((100, 100)), pixel_spacing_um=1.0)
    params = ComparisonParams(cell_size=np.array([25.0, 25.0]))

    origin = _find_grid_origin(surface, params)
    centers = generate_grid_centers(surface, origin, params)

    half = 25.0 / 2
    image_size = 100.0
    in_x = (centers[:, 0] - half < image_size) & (centers[:, 0] + half > 0)
    in_y = (centers[:, 1] - half < image_size) & (centers[:, 1] + half > 0)
    assert np.all(in_x & in_y)


def test_generate_cell_centers_origin_is_first_center():
    """The origin returned by _find_grid_origin appears in the centers list."""
    surface = _make_surface_map(np.ones((60, 60)), pixel_spacing_um=1.0)
    params = ComparisonParams(cell_size=np.array([20.0, 20.0]))

    origin = _find_grid_origin(surface, params)
    centers = generate_grid_centers(surface, origin, params)

    distances = np.linalg.norm(centers - origin, axis=1)
    assert np.any(distances < 1e-9), "origin must appear as one of the cell centers"
