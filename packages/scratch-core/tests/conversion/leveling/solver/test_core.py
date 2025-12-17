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
    fitted_surface, _ = fit_surface(xs=xs, ys=ys, zs=zs, terms=terms)
    leveled_map = zs - fitted_surface
    assert np.var(fitted_surface) < np.var(zs)
    assert np.var(fitted_surface) < np.var(leveled_map)


@pytest.mark.parametrize("terms", list(SurfaceTerms.PLANE))
def test_fit_surface_plane_reduces_variance(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    terms: SurfaceTerms,
):
    leveled_map_terms = zs - fit_surface(xs=xs, ys=ys, zs=zs, terms=terms)[0]
    leveled_map_plane = (
        zs - fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.PLANE)[0]
    )

    single_term_var = np.var(leveled_map_terms)
    plane_var = np.var(leveled_map_plane)

    assert plane_var < single_term_var or np.isclose(plane_var, single_term_var)


@pytest.mark.parametrize("terms", SURFACE_TERMS + [SurfaceTerms.PLANE])
def test_fit_surface_sphere_reduces_variance(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    terms: SurfaceTerms,
):
    leveled_map_terms = zs - fit_surface(xs=xs, ys=ys, zs=zs, terms=terms)[0]
    leveled_map_sphere = (
        zs - fit_surface(xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.SPHERE)[0]
    )

    assert np.var(leveled_map_sphere) < np.var(leveled_map_terms)


@pytest.mark.parametrize("terms", ALL_TERMS)
def test_fit_surface_fits_terms(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    terms: SurfaceTerms,
):
    _, physical_params = fit_surface(xs=xs, ys=ys, zs=zs, terms=terms)
    assert all(not np.isclose(v, 0.0) for p, v in physical_params.items() if p in terms)
    assert all(np.isclose(v, 0.0) for p, v in physical_params.items() if p not in terms)


def test_fit_surface_offset_is_invariant_for_tilted_surfaces(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
):
    _, physical_params_tilt_x = fit_surface(
        xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.TILT_X
    )
    _, physical_params_tilt_y = fit_surface(
        xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.TILT_Y
    )
    assert np.isclose(
        physical_params_tilt_x[SurfaceTerms.OFFSET],
        physical_params_tilt_y[SurfaceTerms.OFFSET],
    )


def test_fit_surface_offset_equals_mean(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
):
    fitted_surface_offset, physical_params_offset = fit_surface(
        xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.OFFSET
    )
    param = physical_params_offset[SurfaceTerms.OFFSET]
    assert np.isclose(np.mean(zs), param)
    assert np.allclose(fitted_surface_offset, param)


def test_fit_surface_defocus_is_positive(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
):
    fitted_surface_defocus, _ = fit_surface(
        xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.DEFOCUS
    )
    assert np.all(fitted_surface_defocus > 0.0)


def test_fit_surface_none_has_no_effect(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
):
    fitted_surface_none, physical_params_none = fit_surface(
        xs=xs, ys=ys, zs=zs, terms=SurfaceTerms.NONE
    )
    assert np.allclose(fitted_surface_none, 0.0)
    assert all(np.isclose(v, 0.0) for v in physical_params_none.values())
