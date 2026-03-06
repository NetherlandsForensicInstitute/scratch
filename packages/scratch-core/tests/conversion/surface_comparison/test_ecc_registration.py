"""
Tests for ECC fine-registration convergence in the surface comparison pipeline.

These tests verify that the ECC algorithm correctly recovers known rigid-body
displacements (translation and rotation) applied to a reference surface.

Surface construction
--------------------
The comparison surface is built from the reference by:
  1. Rotating with scipy.ndimage.rotate(angle_deg) – note that a scipy rotation
     of +θ produces a consensus_rotation of -θ from the pipeline (sign convention).
  2. Applying np.roll(dy, axis=0) / np.roll(dx, axis=1) for pixel translation,
     then filling the wrapped border with low-amplitude noise to avoid artefacts.

The test asserts that the pipeline recovers:
  - consensus_rotation ≈ -scipy_rotation_deg  (atol = 0.01°)
  - consensus_translation ≈ (dx_pix µm, dy_pix µm)  (atol = 0.1 µm)
"""

import numpy as np
import pytest
from scipy.ndimage import rotate as ndrotate

from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams
from conversion.surface_comparison.pipeline import run_comparison_pipeline

# ── tolerances ────────────────────────────────────────────────────────────────
ANGLE_ATOL_DEG = 0.01  # °  – ECC fine-refines to sub-0.01° accuracy
TRANS_ATOL_M = 1e-7  # m  – sub-0.1 µm at 1 µm/px


# ── shared helpers ─────────────────────────────────────────────────────────────


def _reference_surface(seed: int = 7) -> np.ndarray:
    """200×200 surface with rich spatial frequency content."""
    y, x = np.mgrid[0:200, 0:200]
    return (x - y) * np.tanh(x + y / 5) * np.sin(y / 5)


def _make_scan_image(data: np.ndarray, pixel_spacing_m: float = 1e-6) -> ScanImage:
    return ScanImage(data=data, scale_x=pixel_spacing_m, scale_y=pixel_spacing_m)


def _apply_rotation(data: np.ndarray, scipy_angle_deg: float) -> np.ndarray:
    """Rotate using scipy (counter-clockwise in image coords)."""
    return ndrotate(data, scipy_angle_deg, reshape=False, order=3, mode="nearest")


def _apply_translation(
    data: np.ndarray, dy_pix: int, dx_pix: int, noise_rng: np.random.Generator
) -> np.ndarray:
    """Shift with np.roll; fill the wrapped border with low-amplitude noise."""
    shifted = np.roll(np.roll(data, dy_pix, axis=0), dx_pix, axis=1)
    noise = noise_rng.standard_normal(data.shape) * data.std() * 0.05
    result = shifted.copy()
    if dy_pix > 0:
        result[:dy_pix, :] = noise[:dy_pix, :]
    elif dy_pix < 0:
        result[dy_pix:, :] = noise[dy_pix:, :]
    if dx_pix > 0:
        result[:, :dx_pix] = noise[:, :dx_pix]
    elif dx_pix < 0:
        result[:, dx_pix:] = noise[:, dx_pix:]
    return result


def _default_params() -> ComparisonParams:
    return ComparisonParams(
        cell_size=(40e-6, 40e-6),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )


# ── translation-only tests ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dy_pix,dx_pix",
    [
        (3, 5),  # original passing test – kept for regression
        (5, 0),  # pure row shift
        (0, 4),  # pure column shift
        (-4, 3),  # negative row, positive column
        (3, -4),  # positive row, negative column
        (-3, -5),  # both negative
        (6, 2),  # larger row shift
    ],
)
def test_ecc_recovers_translation(dy_pix, dx_pix):
    """ECC converges to the correct translation for a range of pixel offsets."""
    ref_data = _reference_surface()
    rng = np.random.default_rng(seed=abs(dy_pix * 31 + dx_pix))
    comp_data = _apply_translation(ref_data, dy_pix, dx_pix, rng)

    result = run_comparison_pipeline(
        _make_scan_image(ref_data),
        _make_scan_image(comp_data),
        _default_params(),
    )

    tx, ty = result.consensus_translation
    assert abs(tx - dx_pix * 1e-6) <= TRANS_ATOL_M, (
        f"dx={dx_pix}px: expected tx≈{dx_pix}µm, got {tx * 1e6:.4f}µm"
    )
    assert abs(ty - dy_pix * 1e-6) <= TRANS_ATOL_M, (
        f"dy={dy_pix}px: expected ty≈{dy_pix}µm, got {ty * 1e6:.4f}µm"
    )


# ── rotation-only tests ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "scipy_angle_deg",
    [
        -1.5,  # near the search boundary, negative
        -0.8,
        -0.3,
        0.3,
        0.8,
        1.5,  # near the search boundary, positive
    ],
)
def test_ecc_recovers_rotation(scipy_angle_deg):
    """ECC converges to the correct rotation angle.

    Sign convention: a scipy rotation of +θ produces consensus_rotation = -θ.
    """
    ref_data = _reference_surface()
    comp_data = _apply_rotation(ref_data, scipy_angle_deg)

    result = run_comparison_pipeline(
        _make_scan_image(ref_data),
        _make_scan_image(comp_data),
        _default_params(),
    )

    expected_deg = -scipy_angle_deg
    assert abs(result.consensus_rotation - expected_deg) <= ANGLE_ATOL_DEG, (
        f"scipy_angle={scipy_angle_deg}°: expected consensus_rotation≈{expected_deg:.1f}°, "
        f"got {result.consensus_rotation:.4f}°"
    )
    # A pure rotation about the map centre must produce no spurious translation.
    tx, ty = result.consensus_translation
    assert abs(tx) <= TRANS_ATOL_M, f"Pure rotation: unexpected tx={tx * 1e6:.4f}µm"
    assert abs(ty) <= TRANS_ATOL_M, f"Pure rotation: unexpected ty={ty * 1e6:.4f}µm"


# ── combined rotation + translation tests ────────────────────────────────────


@pytest.mark.parametrize(
    "scipy_angle_deg,dy_pix,dx_pix",
    [
        (-1.2, 3, -4),
        (0.6, -3, 5),
        (-0.5, 4, 4),
        (1.0, -2, -6),
    ],
)
def test_ecc_recovers_combined(scipy_angle_deg, dy_pix, dx_pix):
    """ECC correctly recovers both rotation and translation simultaneously."""
    ref_data = _reference_surface()
    rng = np.random.default_rng(
        seed=abs(dy_pix * 17 + dx_pix + int(scipy_angle_deg * 100))
    )
    rotated = _apply_rotation(ref_data, scipy_angle_deg)
    comp_data = _apply_translation(rotated, dy_pix, dx_pix, rng)

    result = run_comparison_pipeline(
        _make_scan_image(ref_data),
        _make_scan_image(comp_data),
        _default_params(),
    )

    expected_rot = -scipy_angle_deg
    tx, ty = result.consensus_translation
    assert abs(result.consensus_rotation - expected_rot) <= ANGLE_ATOL_DEG, (
        f"Expected rotation≈{expected_rot:.1f}°, got {result.consensus_rotation:.4f}°"
    )
    assert abs(tx - dx_pix * 1e-6) <= TRANS_ATOL_M, (
        f"Expected tx≈{dx_pix}µm, got {tx * 1e6:.4f}µm"
    )
    assert abs(ty - dy_pix * 1e-6) <= TRANS_ATOL_M, (
        f"Expected ty≈{dy_pix}µm, got {ty * 1e6:.4f}µm"
    )


# ── identity tests ────────────────────────────────────────────────────────────


def test_ecc_identity_zero_rotation():
    """Identical surfaces produce consensus_rotation ≈ 0°."""
    surface = _make_scan_image(_reference_surface())
    result = run_comparison_pipeline(surface, surface, _default_params())
    assert abs(result.consensus_rotation) <= ANGLE_ATOL_DEG, (
        f"Expected rotation≈0°, got {result.consensus_rotation:.4f}°"
    )


def test_ecc_identity_zero_translation():
    """Identical surfaces produce consensus_translation ≈ (0, 0)."""
    surface = _make_scan_image(_reference_surface())
    result = run_comparison_pipeline(surface, surface, _default_params())
    tx, ty = result.consensus_translation
    assert abs(tx) <= TRANS_ATOL_M, f"Expected tx≈0, got {tx * 1e6:.4f}µm"
    assert abs(ty) <= TRANS_ATOL_M, f"Expected ty≈0, got {ty * 1e6:.4f}µm"
