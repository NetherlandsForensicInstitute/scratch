import numpy as np

from conversion.surface_comparison.cmc_classification import classify_congruent_cells
from conversion.surface_comparison.grid import _find_grid_origin
from conversion.surface_comparison.models import (
    CellResult,
    ComparisonParams,
    ComparisonResult,
    SurfaceMap,
)


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


def _make_params(
    cell_size_um: float, minimum_fill_fraction: float = 0.5
) -> ComparisonParams:
    return ComparisonParams(
        cell_size=np.array([cell_size_um, cell_size_um]),
        minimum_fill_fraction=minimum_fill_fraction,
    )


# ---------------------------------------------------------------------------
# classify_congruent_cells
# ---------------------------------------------------------------------------


def test_classify_cmc_cells_consensus():
    """Two inlier cells and one positional outlier: only the inliers become CMC.

    All registration angles are close to zero, so the consensus angle is 0 and
    _rotate_points is the identity.  Position residuals are [5,5], [5,5], [60,60];
    their median translation is [5,5].  Cell 2's position error [55,55] exceeds
    position_threshold=2, so it is not classified as a CMC.

    Cell 2 also has registration_angle=0.5 rad, which is far outside the default
    angle_threshold of 2 degrees, providing a second independent reason for
    rejection.
    """
    cells = [
        CellResult(np.array([0.0, 0.0]), np.array([5.0, 5.0]), 0.0, 0.8, 1.0),
        CellResult(np.array([10.0, 10.0]), np.array([15.0, 15.0]), 0.0, 0.8, 1.0),
        CellResult(np.array([20.0, 20.0]), np.array([80.0, 80.0]), 0.5, 0.8, 1.0),
    ]
    result = ComparisonResult(cells=cells)
    params = ComparisonParams(correlation_threshold=0.5, position_threshold=2.0)
    reference_center = np.array([0.0, 0.0])

    classify_congruent_cells(result, params, reference_center)

    assert result.congruent_matching_cells_count == 2
    assert cells[0].is_congruent is True
    assert cells[1].is_congruent is True
    assert cells[2].is_congruent is False
    assert np.allclose(result.consensus_translation, [5.0, 5.0])


def test_classify_cmc_score_below_threshold():
    """A cell whose ACCF score is below correlation_threshold is not CMC,
    even when its position and angle agree perfectly with the consensus."""
    cells = [
        CellResult(np.array([0.0, 0.0]), np.array([5.0, 5.0]), 0.0, 0.8, 1.0),
        CellResult(np.array([10.0, 10.0]), np.array([15.0, 5.0]), 0.0, 0.3, 1.0),
    ]
    result = ComparisonResult(cells=cells)
    params = ComparisonParams(correlation_threshold=0.5, position_threshold=50.0)
    reference_center = np.array([0.0, 0.0])

    classify_congruent_cells(result, params, reference_center)

    assert cells[0].is_congruent is True
    assert cells[1].is_congruent is False


def test_classify_cmc_empty_cells():
    """Empty cell list: function returns without error and count stays zero."""
    result = ComparisonResult(cells=[])
    params = ComparisonParams()
    reference_center = np.array([0.0, 0.0])

    classify_congruent_cells(result, params, reference_center)

    assert result.congruent_matching_cells_count == 0


def test_classify_cmc_all_congruent():
    """All four cells agree perfectly on translation: all become CMC."""
    cells = [
        CellResult(np.array([0.0, 0.0]), np.array([5.0, 5.0]), 0.0, 0.9, 1.0),
        CellResult(np.array([10.0, 0.0]), np.array([15.0, 5.0]), 0.0, 0.9, 1.0),
        CellResult(np.array([0.0, 10.0]), np.array([5.0, 15.0]), 0.0, 0.9, 1.0),
        CellResult(np.array([10.0, 10.0]), np.array([15.0, 15.0]), 0.0, 0.9, 1.0),
    ]
    result = ComparisonResult(cells=cells)
    params = ComparisonParams(correlation_threshold=0.5, position_threshold=1.0)
    reference_center = np.array([5.0, 5.0])

    classify_congruent_cells(result, params, reference_center)

    assert result.congruent_matching_cells_count == 4
    assert all(c.is_congruent for c in cells)


# ---------------------------------------------------------------------------
# _find_grid_origin
# ---------------------------------------------------------------------------


def test_find_grid_origin_all_valid():
    """Fully valid 4×4 image with a 4×4 µm cell at 1 µm/px.

    Only one tiling position is possible (n_tiles=1×1).  The single offset
    (oy=0, ox=0) is selected, giving:

        first_center_px = [0 − (4/2 − 0.5),  0 − (4/2 − 0.5)] = [−1.5, −1.5]
        origin_physical = [−1.5, −1.5] × 1 µm/px              = [−1.5, −1.5] µm

    Note: with the full tiling the winning offset is actually (oy=3,ox=3),
    which gives first_center_px = [1.5, 1.5].
    """
    height_map = np.ones((4, 4))
    surface = _make_surface_map(height_map, pixel_spacing_um=1.0)
    params = _make_params(cell_size_um=4.0)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [1.5, 1.5])


def test_find_grid_origin_favors_valid_data_region():
    """Valid data only in the top-left 2×2 corner of a 4×4 image.

    The algorithm must place the origin so that the cell lands on the valid
    data, not on the NaN region.
    Expected: origin = [0.5, −0.5] µm  (verified analytically).
    """
    height_map = np.full((4, 4), np.nan)
    height_map[:2, :2] = 1.0
    surface = _make_surface_map(height_map, pixel_spacing_um=1.0)
    params = _make_params(cell_size_um=2.0)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [0.5, -0.5])


def test_find_grid_origin_tie_sum_equal_min_first_wins():
    """Sum-score and min-score ties: the first candidate in ravel order wins.

    Image: 1 row × 6 cols, NaN at columns 2 and 5 → valid pattern [1,1,0,1,1,0].
    Cell size: 3×1 px at 1 µm/px.

    Offsets ox=1 and ox=2 both tile to two cells with fill 2/3, giving
    identical sum (4/3) and identical minimum (2/3).  Neither criterion
    distinguishes them, so the algorithm picks the first in ravel order (ox=1):

        first_center_px = [1 − (3/2 − 0.5),  0 − (1/2 − 0.5)] = [0.0, 0.0]
        origin_physical = [0.0, 0.0] µm
    """
    height_map = np.ones((1, 6))
    height_map[0, 2] = np.nan
    height_map[0, 5] = np.nan
    surface = _make_surface_map(height_map, pixel_spacing_um=1.0)
    params = _make_params(cell_size_um=3.0)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [0.0, 0.0])


def test_find_grid_origin_surface_too_small_returns_zeros():
    """Image smaller than one cell: fallback to [0, 0]."""
    height_map = np.ones((2, 2))
    surface = _make_surface_map(height_map, pixel_spacing_um=1.0)
    params = _make_params(cell_size_um=10.0)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [0.0, 0.0])


def test_find_grid_origin_all_nan_returns_zeros():
    """All-NaN image: no offset passes minimum_fill_fraction, fallback to [0, 0]."""
    height_map = np.full((6, 6), np.nan)
    surface = _make_surface_map(height_map, pixel_spacing_um=1.0)
    params = _make_params(cell_size_um=3.0)

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [0.0, 0.0])


def test_find_grid_origin_pixel_spacing_scales_result():
    """Physical origin scales proportionally with pixel_spacing.

    A 4×4 image with 2 µm/px spacing and a 4-px (=8 µm) cell gives the same
    pixel-space offset as the 1 µm/px case, but the physical origin is doubled:

        first_center_px = [1.5, 1.5] px  (same winner as test_find_grid_origin_all_valid)
        origin_physical = [1.5, 1.5] × 2 µm/px = [3.0, 3.0] µm
    """
    height_map = np.ones((4, 4))
    surface = _make_surface_map(height_map, pixel_spacing_um=2.0)
    params = ComparisonParams(
        cell_size=np.array([8.0, 8.0]),  # 8 µm / 2 µm/px = 4 px
        minimum_fill_fraction=0.5,
    )

    origin = _find_grid_origin(surface, params)

    assert np.allclose(origin, [3.0, 3.0])
