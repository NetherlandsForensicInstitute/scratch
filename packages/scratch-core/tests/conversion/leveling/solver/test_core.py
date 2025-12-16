from conversion.leveling import SurfaceTerms
from conversion.leveling.solver import fit_surface
import numpy as np
from numpy.typing import NDArray
import pytest

from ..constants import ALL_TERMS, SURFACE_TERMS


@pytest.mark.parametrize("terms", ALL_TERMS)
def test_fit_surface_reduces_variance(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    terms: SurfaceTerms,
):
    result = fit_surface(xs=xs, ys=ys, zs=zs, terms=terms)
    fitted = result.fitted_surface
    highpass = zs - result.fitted_surface
    assert np.var(fitted) < np.var(zs)
    assert np.var(fitted) < np.var(highpass)


@pytest.mark.parametrize("terms", list(SurfaceTerms.PLANE))
def test_fit_surface_plane_reduces_variance(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    terms: SurfaceTerms,
):
    single_term = zs - fit_surface(xs=xs, ys=ys, zs=zs, terms=terms).fitted_surface
    plane = (
        zs - fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.PLANE).fitted_surface
    )

    single_term_var = np.var(single_term)
    plane_var = np.var(plane)

    assert plane_var < single_term_var or np.isclose(plane_var, single_term_var)


@pytest.mark.parametrize("terms", SURFACE_TERMS + [SurfaceTerms.PLANE])
def test_fit_surface_sphere_reduces_variance(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    terms: SurfaceTerms,
):
    single_term = zs - fit_surface(xs=xs, ys=ys, zs=zs, terms=terms).fitted_surface
    sphere = (
        zs - fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.SPHERE).fitted_surface
    )

    assert np.var(sphere) < np.var(single_term)


@pytest.mark.parametrize("terms", ALL_TERMS)
def test_fit_surface_fits_terms(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    terms: SurfaceTerms,
):
    result = fit_surface(xs=xs, ys=ys, zs=zs, terms=terms)
    params = result.physical_params
    assert all(not np.isclose(v, 0.0) for p, v in params.items() if p in terms)
    assert all(np.isclose(v, 0.0) for p, v in params.items() if p not in terms)


def test_fit_surface_offset_is_invariant_for_tilted_surfaces(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
):
    result_tilt_x = fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.TILT_X)
    result_tilt_y = fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.TILT_Y)
    assert np.isclose(
        result_tilt_x.physical_params[SurfaceTerms.OFFSET],
        result_tilt_y.physical_params[SurfaceTerms.OFFSET],
    )


def test_fit_surface_offset_equals_mean(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
):
    result_offset = fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.OFFSET)
    param = result_offset.physical_params[SurfaceTerms.OFFSET]
    assert np.isclose(np.mean(zs), param)
    assert np.allclose(result_offset.fitted_surface, param)


def test_fit_surface_defocus_is_positive(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
):
    result_defocus = fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.DEFOCUS)
    assert np.all(result_defocus.fitted_surface > 0.0)


def test_fit_surface_none_has_no_effect(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
):
    result_none = fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.NONE)
    assert np.allclose(result_none.fitted_surface, 0.0)
    assert all(np.isclose(v, 0.0) for v in result_none.physical_params.values())
