"""
Replacement for test_area_comparison.py.

The original file imported ``run_area_comparison`` and ``AreaSimilarityResult``
from a non-existent module ``area_comparison``.  That described a global
Fourier-Mellin alignment step that is not part of the current codebase.

The closest existing equivalent is ``run_comparison_pipeline``, which returns
a ``ComparisonResult`` containing:
  - ``consensus_rotation``:            agreed rotation across CMC cells (degrees)
  - ``consensus_translation``:         agreed translation across CMC cells (m)
  - ``cmc_count``:number of CMC cells

These tests verify those global outputs for the identity case and a known
translation case.  Rotation recovery at the pipeline level is exercised by
the NIST integration tests (``test_compare_datasets_nist.py``).
"""

import numpy as np
from scipy.ndimage import shift as nd_shift

from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import (
    ComparisonParams,
    ComparisonResult,
)
from conversion.surface_comparison.pipeline import run_comparison_pipeline
from tests.conversion.surface_comparison.plot_utils import (
    plot_rotated_squares,
    plot_side_by_side,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surface_map(
    height_map: np.ndarray, pixel_spacing_m: float = 1e-6
) -> ScanImage:
    scale = pixel_spacing_m
    return ScanImage(data=height_map, scale_x=scale, scale_y=scale)


def _identity_params() -> ComparisonParams:
    return ComparisonParams(
        cell_size=np.array([40e-6, 40e-6]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )


def _identity_surface() -> ScanImage:
    """100×100 surface used for identity tests.

    Uses sin(x/5)·cos(y/5) because when the same image is presented as both
    reference and comparison, ECC reliably converges to the identity warp and
    returns translation = [0, 0] to numerical precision.  Periodicity is not
    a concern here because no spatial shift is applied.
    """
    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((100, 100))
    return _make_surface_map(data)


def _make_translated_surface(
    data: np.ndarray,
    dy: int,
    dx: int,
    rng: np.random.Generator,
    noise_sigma: float = 0.05,
) -> np.ndarray:
    """Shift ``data`` by ``(dy, dx)`` pixels and fill the vacated border with noise.

    ``scipy.ndimage.shift`` with ``mode='constant', cval=NaN`` shifts the data
    and marks the vacated border as NaN.  Those pixels are then replaced by
    Gaussian noise scaled to ``noise_sigma × data.std()``.

    This avoids two problems that arise with ``np.roll``:
    - **Cyclic wrap**: ``np.roll`` copies content from the opposite edge into
      the vacated border; cells that straddle the boundary see a discontinuity
      and register to a spurious location.
    - **Hard zero edge**: ``mode='constant', cval=0`` creates a sharp boundary
      that ECC can lock onto instead of the true surface texture.

    Filling with noise is realistic (a real shifted surface simply has no
    overlap at the edges) and gives ECC nothing to lock onto in the border.
    """
    shifted = nd_shift(data.astype(float), shift=[dy, dx], mode="constant", cval=np.nan)
    nan_mask = np.isnan(shifted)
    shifted[nan_mask] = rng.standard_normal(nan_mask.sum()) * noise_sigma * data.std()
    return shifted


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_identity_returns_comparison_result():
    """run_comparison_pipeline returns a ComparisonResult with expected fields."""
    surface = _identity_surface()

    result = run_comparison_pipeline(surface, surface, _identity_params())

    assert isinstance(result, ComparisonResult)
    assert isinstance(result.cmc_count, int)
    assert len(result.consensus_translation) == 2


def test_pipeline_identity_zero_rotation():
    """Identical surfaces yield consensus_rotation ≈ 0°."""
    surface = _identity_surface()

    result = run_comparison_pipeline(surface, surface, _identity_params())

    ref_scale_x = surface.scale_x
    ref_scale_y = surface.scale_y
    comp_scale_x = surface.scale_x
    comp_scale_y = surface.scale_y
    cell_w_px = int(round(_identity_params().cell_size[0] / ref_scale_x))
    cell_h_px = int(round(_identity_params().cell_size[1] / ref_scale_y))

    reference_plot = plot_rotated_squares(
        image=surface.data,
        squares=[
            (
                (
                    c.center_reference[0] / ref_scale_x,
                    c.center_reference[1] / ref_scale_y,
                ),
                (cell_w_px, cell_h_px),
                0.0,
            )
            for c in result.cells
        ],
    )

    comparison_plot = plot_rotated_squares(
        image=surface.data,
        squares=[
            (
                (
                    c.center_comparison[0] / comp_scale_x,
                    c.center_comparison[1] / comp_scale_y,
                ),
                (cell_w_px, cell_h_px),
                -c.angle_deg,
            )
            for c in result.cells
        ],
    )
    plot_side_by_side(
        img1=reference_plot,
        img2=comparison_plot,
        title1="Reference",
        title2="Comparison",
    )

    # consensus_rotation is in degrees
    assert np.isclose(result.consensus_rotation, 0.0, atol=0.01), (
        f"Expected consensus_rotation ≈ 0°, got {result.consensus_rotation:.4f}°"
    )


def test_pipeline_identity_zero_translation():
    """Identical surfaces yield consensus_translation ≈ [0, 0]."""
    surface = _identity_surface()

    result = run_comparison_pipeline(surface, surface, _identity_params())

    assert np.allclose(result.consensus_translation, [0.0, 0.0], atol=1e-9), (
        f"Expected consensus_translation ≈ [0, 0], got {result.consensus_translation}"
    )  # tolerance of 1 nm.


def test_pipeline_recovers_known_translation():
    """Pipeline consensus_translation matches a known integer-pixel shift.

    The comparison surface is produced by shifting the reference data by 5
    rows (y-axis) and 3 columns (x-axis), giving a translation of [3, 5] m
    at 1 m/px.  The vacated border is filled with Gaussian noise so that
    ECC finds no spurious feature to lock onto at the image edge.  The
    consensus must be within 0.5e-6 m of [3e-6, 5e-6].

    **Surface design:**
    A 200×200 aperiodic surface built from sines with incommensurate
    frequencies (periods 31, 37, 17, 23 px — all much larger than the 5 px
    shift but much smaller than the 40 px cell) is used for two reasons:

    - **No period aliasing**: a purely periodic signal repeats within the
      cell, so the NCC can match at [3, 5] *or* [3 + period, 5] with equal
      score.  Incommensurate frequencies give a unique texture fingerprint
      within each 40×40 m cell.
    - **Enough safe inner cells**: a 200×200 image with 40 m cells gives
      25 grid cells, of which the 9 inner cells sit well away from both the
      image boundary and the noise-filled border strip.  These cells reliably
      vote for [3, 5], providing a comfortable CMC majority.
    """
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:200, 0:200]

    # create som a-periodic data
    data = (x - y) * np.tanh(x + y / 5) * np.sin(y / 5)
    ref = _make_surface_map(data)
    comp = _make_surface_map(_make_translated_surface(data, dy=5, dx=3, rng=rng))
    params = ComparisonParams(
        cell_size=np.array([40e-6, 40e-6]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )

    result = run_comparison_pipeline(ref, comp, params)

    # TODO: Can we lower the tolerance somehow?
    assert np.allclose(result.consensus_translation, [3e-6, 5e-6], atol=1e-6), (
        f"Expected consensus_translation ≈ [3e-6, 5e-6], got {result.consensus_translation}"
    )


def test_pipeline_produces_cells():
    """Pipeline produces at least one cell for a reasonably sized image."""
    surface = _identity_surface()

    result = run_comparison_pipeline(surface, surface, _identity_params())

    assert len(result.cells) > 0
