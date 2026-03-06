"""
Pipeline-level integration tests for ``run_comparison_pipeline``.

These tests verify the public contract of the pipeline:
  - it returns a ``ComparisonResult`` with the expected structure,
  - it recovers a known integer-pixel translation, and
  - it produces at least one cell for a reasonably sized image.

ECC convergence (rotation, translation, combined, identity) is covered in
depth by ``test_ecc_registration.py``.
"""

import numpy as np
from scipy.ndimage import shift as nd_shift

from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams, ComparisonResult
from conversion.surface_comparison.pipeline import run_comparison_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surface_map(
    height_map: np.ndarray, pixel_spacing_m: float = 1e-6
) -> ScanImage:
    return ScanImage(data=height_map, scale_x=pixel_spacing_m, scale_y=pixel_spacing_m)


def _identity_surface() -> ScanImage:
    """100×100 random surface used as a smoke-test fixture."""
    rng = np.random.default_rng(seed=42)
    return _make_surface_map(rng.standard_normal((100, 100)))


def _identity_params() -> ComparisonParams:
    return ComparisonParams(
        cell_size=np.array([40e-6, 40e-6]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )


def _make_translated_surface(
    data: np.ndarray,
    dy: int,
    dx: int,
    rng: np.random.Generator,
    noise_sigma: float = 0.05,
) -> np.ndarray:
    """Shift ``data`` by ``(dy, dx)`` pixels; fill the vacated border with noise.

    Uses ``scipy.ndimage.shift`` so that the vacated border contains no
    recycled content from the opposite edge (unlike ``np.roll``).
    """
    shifted = nd_shift(data.astype(float), shift=[dy, dx], mode="constant", cval=np.nan)
    nan_mask = np.isnan(shifted)
    shifted[nan_mask] = rng.standard_normal(nan_mask.sum()) * noise_sigma * data.std()
    return shifted


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_identity_returns_comparison_result():
    """``run_comparison_pipeline`` returns a ``ComparisonResult`` with expected fields."""
    surface = _identity_surface()

    result = run_comparison_pipeline(surface, surface, _identity_params())

    assert isinstance(result, ComparisonResult)
    assert isinstance(result.cmc_count, int)
    assert len(result.consensus_translation) == 2


def test_pipeline_produces_cells():
    """Pipeline produces at least one cell for a reasonably sized image."""
    surface = _identity_surface()

    result = run_comparison_pipeline(surface, surface, _identity_params())

    assert len(result.cells) > 0


def test_pipeline_recovers_known_translation():
    """``consensus_translation`` matches a known integer-pixel shift within 1 µm.

    The comparison surface is produced by shifting the reference data by 5 rows
    and 3 columns (→ translation [3 µm, 5 µm] at 1 µm/px).  The vacated border
    is filled with low-amplitude Gaussian noise so that ECC has nothing spurious
    to lock onto at the image edge.
    """
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:200, 0:200]
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

    assert np.allclose(result.consensus_translation, [3e-6, 5e-6], atol=1e-6), (
        f"Expected consensus_translation ≈ [3e-6, 5e-6], got {result.consensus_translation}"
    )
