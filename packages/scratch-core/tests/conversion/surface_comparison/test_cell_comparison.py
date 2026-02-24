"""
Replacement for test_cell_comparison.py.

The original file imported ``run_cell_comparison`` from a non-existent module
``cell_comparison``.  The equivalent public function is ``register_cells`` in
``cell_registration``, which runs the three-stage per-cell registration
pipeline (coarse NCC sweep → phase cross-correlation → ECC refinement).
"""

import numpy as np

from conversion.surface_comparison.cell_registration import register_cells
from conversion.surface_comparison.models import (
    CellResult,
    ComparisonParams,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_cells_identity_scores():
    """Registering a surface against itself yields near-perfect ECC scores.

    ECC converges to the identity warp and returns a score very close to 1.0
    for every cell.  OpenCV's ECC can return values very slightly above 1.0
    due to floating-point arithmetic, so the bound check uses a small tolerance.
    """
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5.0) * np.cos(y / 5.0)
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    cells = register_cells(surface, surface, params)

    assert len(cells) > 0
    for cell in cells:
        assert cell.area_cross_correlation_function_score > 0.99, (
            f"Expected score > 0.99, got {cell.area_cross_correlation_function_score:.6f}"
        )


def test_register_cells_identity_angles():
    """Registering a surface against itself yields registration angles at zero."""
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5.0) * np.cos(y / 5.0)
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    cells = register_cells(surface, surface, params)

    for cell in cells:
        assert abs(np.degrees(cell.registration_angle)) < 0.1, (
            f"Expected angle ≈ 0°, got {np.degrees(cell.registration_angle):.4f}°"
        )


def test_register_cells_returns_valid_cell_results():
    """register_cells returns CellResult objects with well-formed fields.

    ECC scores are expected to be very close to 1.0 but may be fractionally
    above 1.0 due to floating-point arithmetic, so the upper bound uses a
    small tolerance (1.0 + 1e-5) rather than a strict 1.0.
    """
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5.0) * np.cos(y / 5.0)
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    cells = register_cells(surface, surface, params)

    assert len(cells) > 0
    for cell in cells:
        assert isinstance(cell, CellResult)
        assert cell.center_reference.shape == (2,)
        assert cell.center_comparison.shape == (2,)
        assert -1.0 <= cell.area_cross_correlation_function_score <= 1.0 + 1e-5, (
            f"Score {cell.area_cross_correlation_function_score:.8f} outside [-1, 1+eps]"
        )
        assert 0.0 < cell.reference_fill_fraction <= 1.0


def test_register_cells_no_cells_for_tiny_image():
    """An image much smaller than the cell yields no cells.

    A 10×10 image with a 40×40 µm cell at 1 µm/px gives geometric overlap
    of 100/1600 = 6.25 %, which is below any reasonable minimum_fill_fraction.
    """
    data = np.ones((10, 10))
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        minimum_fill_fraction=0.5,
    )

    cells = register_cells(surface, surface, params)

    assert cells == []


def test_register_cells_high_fill_fraction_excludes_partial_cells():
    """Raising minimum_fill_fraction to 0.9 filters out edge cells.

    A 30×30 image with a 40×40 µm cell has only partial geometric overlap
    (at most 30×30 / 40×40 = 56.25 %).  With minimum_fill_fraction=0.9
    no cell passes the threshold.
    """
    data = np.ones((30, 30))
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        minimum_fill_fraction=0.9,
    )

    cells = register_cells(surface, surface, params)

    assert cells == []
