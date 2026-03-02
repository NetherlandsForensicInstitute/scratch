import numpy as np

from conversion.surface_comparison.grid_old import (
    _find_grid_origin,
    generate_grid_centers,
)
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surface_map(
    height_map: np.ndarray, pixel_spacing_m: float = 1e-6
) -> ScanImage:
    scale = pixel_spacing_m
    return ScanImage(data=height_map, scale_x=scale, scale_y=scale)


def _make_params(
    cell_size_m: float, minimum_fill_fraction: float = 0.5
) -> ComparisonParams:
    return ComparisonParams(
        cell_size=np.array([cell_size_m, cell_size_m]),
        minimum_fill_fraction=minimum_fill_fraction,
    )


# ---------------------------------------------------------------------------
# _find_grid_origin
# ---------------------------------------------------------------------------


def test_find_optimal_cell_origin_returns_physical_coordinate():
    """_find_grid_origin returns a 2-element physical coordinate in m.

    For a fully valid 100×100 image with a 20×20 m cell at 1 m/px the
    origin must lie within the image bounds [0, 100) in both axes.
    """
    surface = _make_surface_map(np.ones((100, 100)), pixel_spacing_m=1e-6)
    params = ComparisonParams(cell_size=np.array([20e-6, 20e-6]))

    origin = _find_grid_origin(surface, params)

    assert origin.shape == (2,)
    assert 0.0 <= origin[0] < 100e-6
    assert 0.0 <= origin[1] < 100e-6


def test_find_grid_origin_all_valid():
    """Fully valid 4×4 image with a 4×4 m cell at 1 m/px.

    Only one tiling position is possible (n_tiles=1×1).  The single offset
    (oy=0, ox=0) is selected, giving:

        first_center_px = [0 − (4/2 − 0.5),  0 − (4/2 − 0.5)] = [−1.5, −1.5]
        origin_physical = [−1.5, −1.5] × 1 m/px              = [−1.5, −1.5] m

    Note: with the full tiling the winning offset is actually (oy=3,ox=3),
    which gives first_center_px = [1.5, 1.5].
    """
    height_map = np.ones((4, 4))
    surface = _make_surface_map(height_map, pixel_spacing_m=1e-6)
    params = _make_params(cell_size_m=4e-6)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [1.5e-6, 1.5e-6])


def test_find_grid_origin_favors_valid_data_region():
    """Valid data only in the top-left 2×2 corner of a 4×4 image.

    The algorithm must place the origin so that the cell lands on the valid
    data, not on the NaN region.
    Expected: origin = [0.5, −0.5] m  (verified analytically).
    """
    height_map = np.full((4, 4), np.nan)
    height_map[:2, :2] = 1.0
    surface = _make_surface_map(height_map, pixel_spacing_m=1e-6)
    params = _make_params(cell_size_m=2e-6)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [0.5e-6, -0.5e-6])


def test_find_grid_origin_tie_sum_equal_min_first_wins():
    """Sum-score and min-score ties: the first candidate in ravel order wins.

    Image: 1 row × 6 cols, NaN at columns 2 and 5 → valid pattern [1,1,0,1,1,0].
    Cell size: 3×1 px at 1 m/px.

    Offsets ox=1 and ox=2 both tile to two cells with fill 2/3, giving
    identical sum (4/3) and identical minimum (2/3).  Neither criterion
    distinguishes them, so the algorithm picks the first in ravel order (ox=1):

        first_center_px = [1 − (3/2 − 0.5),  0 − (1/2 − 0.5)] = [0.0, 0.0]
        origin_physical = [0.0, 0.0] m
    """
    height_map = np.ones((1, 6))
    height_map[0, 2] = np.nan
    height_map[0, 5] = np.nan
    surface = _make_surface_map(height_map, pixel_spacing_m=1e-6)
    params = _make_params(cell_size_m=3e-6)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [0.0, 0.0])


def test_find_grid_origin_surface_too_small_returns_zeros():
    """Image smaller than one cell: fallback to [0, 0]."""
    height_map = np.ones((2, 2))
    surface = _make_surface_map(height_map, pixel_spacing_m=1e-6)
    params = _make_params(cell_size_m=10e-6)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [0.0, 0.0])


def test_find_grid_origin_all_nan_returns_zeros():
    """All-NaN image: no offset passes minimum_fill_fraction, fallback to [0, 0]."""
    height_map = np.full((6, 6), np.nan)
    surface = _make_surface_map(height_map, pixel_spacing_m=1e-6)
    params = _make_params(cell_size_m=3e-6)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [0.0, 0.0])


def test_find_grid_origin_pixel_spacing_scales_result():
    """Physical origin scales proportionally with pixel_spacing.

    A 4×4 image with 2 m/px spacing and a 4-px (=8 m) cell gives the same
    pixel-space offset as the 1 m/px case, but the physical origin is doubled:

        first_center_px = [1.5, 1.5] px  (same winner as test_find_grid_origin_all_valid)
        origin_physical = [1.5, 1.5] × 2 m/px = [3.0, 3.0] m
    """
    height_map = np.ones((4, 4))
    surface = _make_surface_map(height_map, pixel_spacing_m=2e-6)
    params = ComparisonParams(
        cell_size=np.array([8e-6, 8e-6]),  # 8 m / 2 m/px = 4 px
        minimum_fill_fraction=0.5,
    )

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [3e-6, 3e-6])


# ---------------------------------------------------------------------------
# generate_grid_centers  (not covered elsewhere)
# ---------------------------------------------------------------------------


def test_generate_cell_centers_spacing():
    """Adjacent centers are separated by exactly cell_size in both axes."""
    surface = _make_surface_map(np.ones((100, 100)), pixel_spacing_m=1e-6)
    params = ComparisonParams(cell_size=np.array([25e-6, 25e-6]))

    origin = _find_grid_origin(surface, params)
    centers = generate_grid_centers(surface, origin, params)

    assert len(centers) > 0
    x_unique = np.unique(centers[:, 0])
    y_unique = np.unique(centers[:, 1])
    assert np.allclose(np.diff(x_unique), 25e-6)
    assert np.allclose(np.diff(y_unique), 25e-6)


def test_generate_cell_centers_all_overlap_image():
    """Every returned cell at least partially overlaps the reference image.

    _axis_centers extends the grid until cells fall *entirely* outside the
    image, so cells whose edge just reaches the image boundary are included.
    The correct check is therefore that each cell's footprint overlaps the
    image, not that the center itself lies within it.
    """
    surface = _make_surface_map(np.ones((100, 100)), pixel_spacing_m=1e-6)
    params = ComparisonParams(cell_size=np.array([25e-6, 25e-6]))

    origin = _find_grid_origin(surface, params)
    centers = generate_grid_centers(surface, origin, params)

    half = 25e-6 / 2
    image_size = 100e-6
    in_x = (centers[:, 0] - half < image_size) & (centers[:, 0] + half > 0)
    in_y = (centers[:, 1] - half < image_size) & (centers[:, 1] + half > 0)
    assert np.all(in_x & in_y)


def test_generate_cell_centers_origin_is_first_center():
    """The origin returned by _find_grid_origin appears in the centers list."""
    surface = _make_surface_map(np.ones((60, 60)), pixel_spacing_m=1e-6)
    params = ComparisonParams(cell_size=np.array([20e-6, 20e-6]))

    origin = _find_grid_origin(surface, params)
    centers = generate_grid_centers(surface, origin, params)

    distances = np.linalg.norm(centers - origin, axis=1)
    assert np.any(distances < 1e-9), "origin must appear as one of the cell centers"
