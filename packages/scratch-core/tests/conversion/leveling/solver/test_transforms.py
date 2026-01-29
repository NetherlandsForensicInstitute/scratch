import numpy as np
import pytest

from container_models.base import FloatArray1D
from conversion.leveling import SurfaceTerms
from conversion.leveling.solver import normalize_coordinates, denormalize_parameters
from ..constants import SINGLE_TERMS


class TestNormalizeCoordinates:
    def test_normalized_coordinates_have_zero_mean(
        self, xs: FloatArray1D, ys: FloatArray1D
    ):
        normalized = normalize_coordinates(xs=xs, ys=ys)
        assert np.isclose(np.mean(normalized.xs), 0.0)
        assert np.isclose(np.mean(normalized.ys), 0.0)
        assert normalized.xs[0] < 0 < normalized.xs[-1]
        assert normalized.ys[0] < 0 < normalized.ys[-1]

    def test_normalized_coordinates_are_strictly_increasing(
        self, xs: FloatArray1D, ys: FloatArray1D
    ):
        normalized = normalize_coordinates(xs=xs, ys=ys)
        assert np.all(normalized.xs[:-1] < normalized.xs[1:])
        assert np.all(normalized.ys[:-1] < normalized.ys[1:])

    def test_normalize_coordinates_returns_means(
        self, xs: FloatArray1D, ys: FloatArray1D
    ):
        normalized = normalize_coordinates(xs=xs, ys=ys)
        assert np.isclose(normalized.x_mean, np.mean(xs))
        assert np.isclose(normalized.y_mean, np.mean(ys))

    def test_normalized_coordinates_are_bounded_by_unit_disk(
        self, xs: FloatArray1D, ys: FloatArray1D
    ):
        normalized = normalize_coordinates(xs=xs, ys=ys)
        assert np.isclose(normalized.xs[-1] - normalized.xs[0], 1.0)
        assert np.isclose(normalized.ys[-1] - normalized.ys[0], 0.5)


class TestDenormalizeParameters:
    @pytest.mark.parametrize("terms", SINGLE_TERMS)
    def test_denormalize_parameters_returns_all_terms(self, terms: SurfaceTerms):
        params = denormalize_parameters(
            coefficients={terms: 1.5}, x_mean=0.1, y_mean=0.5, scale=0.21
        )
        assert all(term in params for term in SurfaceTerms)

    @pytest.mark.parametrize("terms", SINGLE_TERMS)
    def test_denormalize_parameters_adjusts_terms(self, terms: SurfaceTerms):
        initial_value = 1.5
        params = denormalize_parameters(
            coefficients={terms: initial_value}, x_mean=0.1, y_mean=0.5, scale=0.21
        )
        assert all(
            not np.isclose(params[t], initial_value)
            for t in SurfaceTerms
            if t in terms and t != SurfaceTerms.OFFSET
        )

    def test_denormalize_parameters_matches_baseline_output(self):
        params = denormalize_parameters(
            coefficients={term: 1.0 for term in SurfaceTerms},
            x_mean=0.1,
            y_mean=0.5,
            scale=0.21,
        )
        assert np.isclose(params[SurfaceTerms.OFFSET], 0.877087)
        assert np.isclose(params[SurfaceTerms.TILT_X], 0.17031)
        assert np.isclose(params[SurfaceTerms.TILT_Y], 0.20559)
        assert np.isclose(params[SurfaceTerms.ASTIG_45], 0.0441)
        assert np.isclose(params[SurfaceTerms.DEFOCUS], 0.0441)
        assert np.isclose(params[SurfaceTerms.ASTIG_0], 0.0441)
