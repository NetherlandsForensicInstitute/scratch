"""
Replacement for test_area_comparison.py.

The original file imported ``run_area_comparison`` and ``AreaSimilarityResult``
from a non-existent module ``area_comparison``.  That described a global
Fourier-Mellin alignment step that is not part of the current codebase.

The closest existing equivalent is ``run_comparison_pipeline``, which returns
a ``ComparisonResult`` containing:
  - ``consensus_rotation``:            agreed rotation across CMC cells (radians)
  - ``consensus_translation``:         agreed translation across CMC cells (µm)
  - ``congruent_matching_cells_count``:number of CMC cells

These tests verify those global outputs for the identity case and a known
translation case.  Rotation recovery at the pipeline level is exercised by
the NIST integration tests (``test_compare_datasets_nist.py``).
"""

import numpy as np

from conversion.surface_comparison.models import (
    ComparisonParams,
    ComparisonResult,
    SurfaceMap,
)
from conversion.surface_comparison.pipeline import run_comparison_pipeline


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


def test_pipeline_identity_returns_comparison_result():
    """run_comparison_pipeline returns a ComparisonResult with expected fields."""
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5.0) * np.cos(y / 5.0)
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    result = run_comparison_pipeline(surface, surface, params)

    assert isinstance(result, ComparisonResult)
    assert isinstance(result.congruent_matching_cells_count, int)
    assert result.consensus_translation.shape == (2,)


def test_pipeline_identity_zero_rotation():
    """Identical surfaces yield consensus_rotation ≈ 0."""
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5.0) * np.cos(y / 5.0)
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    result = run_comparison_pipeline(surface, surface, params)

    assert np.isclose(result.consensus_rotation, 0.0, atol=np.radians(0.01)), (
        f"Expected consensus_rotation ≈ 0°, "
        f"got {np.degrees(result.consensus_rotation):.4f}°"
    )


def test_pipeline_identity_zero_translation():
    """Identical surfaces yield consensus_translation ≈ [0, 0]."""
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5.0) * np.cos(y / 5.0)
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    result = run_comparison_pipeline(surface, surface, params)

    assert np.allclose(result.consensus_translation, [0.0, 0.0], atol=1e-5), (
        f"Expected consensus_translation ≈ [0, 0], got {result.consensus_translation}"
    )


def test_pipeline_recovers_known_translation():
    """Pipeline consensus_translation matches a known integer-pixel shift.

    The comparison surface is produced by np.roll, which cyclically shifts
    the data by 5 rows (y-axis) and 3 columns (x-axis), giving a translation
    of [3, 5] µm at 1 µm/px.  The consensus must be within 0.5 µm of this.

    np.roll(shift=5, axis=0) moves rows downward → comparison is displaced
    +5 µm in y relative to reference.
    np.roll(shift=3, axis=1) moves columns rightward → comparison is displaced
    +3 µm in x relative to reference.
    """
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5.0) * np.cos(y / 5.0)
    ref = _make_surface_map(data)
    comp = _make_surface_map(np.roll(data, (5, 3), axis=(0, 1)))
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    result = run_comparison_pipeline(ref, comp, params)

    assert np.allclose(result.consensus_translation, [3.0, 5.0], atol=0.5), (
        f"Expected consensus_translation ≈ [3, 5], got {result.consensus_translation}"
    )


def test_pipeline_produces_cells():
    """Pipeline produces at least one cell for a reasonably sized image."""
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5.0) * np.cos(y / 5.0)
    surface = _make_surface_map(data)
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    result = run_comparison_pipeline(surface, surface, params)

    assert len(result.cells) > 0
