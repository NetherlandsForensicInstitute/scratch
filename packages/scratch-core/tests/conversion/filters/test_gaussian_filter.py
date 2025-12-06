"""Tests for Gaussian filter."""

from collections.abc import Callable
from typing import Final, TypeAlias
from functools import partial
import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.constants import deci, femto

from conversion.filters.gaussian_filter import (
    apply_gaussian_filter,
    get_cutoff_sigmas,
    get_alpha,
)
from conversion.filters.protocol import FilterFlags
from utils.array_definitions import ScanMap2DArray

CUTOFF: Final[float] = 5.0
SEED: Final[int] = 42
approx = partial(pytest.approx, rel=femto)
allclose = partial(np.allclose, rtol=femto)
Sigma: TypeAlias = Callable[[NDArray[np.floating]], NDArray[np.floating]]


@pytest.fixture(scope="module")
def alpha() -> float:
    return get_alpha(regression_order=0)


@pytest.fixture(scope="module")
def sigma(alpha) -> Sigma:
    return lambda cutoffs: get_cutoff_sigmas(alpha, cutoff_lengths=cutoffs)


@pytest.fixture(scope="class")
def cutoff_pixels() -> NDArray[np.floating]:
    """Default cutoff pixels for Gaussian filter tests."""
    return np.array([3.0, 3.0])


@pytest.fixture(autouse=True)
def reset_random_seed() -> None:
    """Reset numpy random seed before each test for reproducibility."""
    np.random.seed(SEED)


class TestCutoffToSigma:
    def test_conversion_formula(self, sigma: Sigma) -> None:
        # arrange
        cutoffs = np.arange(1.0, 6.0)
        expected = np.array(
            [
                0.1873906251292776,
                0.3747812502585552,
                0.5621718753878329,
                0.7495625005171104,
                0.936953125646388,
            ]
        )
        # act
        sigma_ = sigma(cutoffs)

        # assert
        assert sigma_ == approx(expected)
        assert (sigma_ < cutoffs).all()

    def test_linear_scaling(self, sigma: Sigma) -> None:
        # arrange
        delta = np.array([5, 10, 15, 20, 25])
        cutoffs = np.array([2, 4, 8, 16, 32])

        # act and assert
        assert sigma(delta * cutoffs) == approx(delta * sigma(cutoffs))

    def test_zero_cutoffs(self, alpha: float) -> None:
        # arrange
        cutoffs = np.array([0])

        # act
        sigma = get_cutoff_sigmas(alpha, cutoffs)

        # assert
        assert np.array_equal(cutoffs, sigma)


class TestGaussianFilterFunction:
    """Test the gaussian_filter function."""

    @pytest.fixture(scope="class")
    def array_data(self) -> ScanMap2DArray:
        data = np.full((10, 10), 5.0)
        data[5, 5] = np.nan
        return data

    @pytest.mark.parametrize(
        "flags, expected",
        (
            pytest.param(
                FilterFlags.NAN_OUT & ~FilterFlags.HIGH_PASS,
                True,
                id="with NAN_OUT flag",
            ),
            pytest.param(
                ~(FilterFlags.NAN_OUT | FilterFlags.HIGH_PASS),
                False,
                id="without NAN_OUT flag",
            ),
        ),
    )
    def test_nan_out_flag(
        self,
        flags: FilterFlags,
        expected: float,
        alpha: float,
        array_data: ScanMap2DArray,
        cutoff_pixels: NDArray[np.floating],
    ) -> None:
        # act
        result = apply_gaussian_filter(
            array_data,
            alpha,
            cutoff_pixels=cutoff_pixels,
            flags=flags,
        )

        # assert
        assert np.isnan(result).any() == expected

    @pytest.mark.parametrize(
        "flags, expected",
        (
            pytest.param(
                ~(FilterFlags.NAN_OUT | FilterFlags.HIGH_PASS),
                5.0,
                id="without highpass flag",
            ),
            pytest.param(
                ~FilterFlags.NAN_OUT & FilterFlags.HIGH_PASS,
                0.0,
                id="with highpass flag",
            ),
        ),
    )
    def test_high_pass_flag(
        self,
        flags: FilterFlags,
        expected: float,
        alpha: float,
        array_data: ScanMap2DArray,
        cutoff_pixels: NDArray[np.floating],
    ) -> None:
        # act
        result = apply_gaussian_filter(
            array_data,
            alpha,
            cutoff_pixels=cutoff_pixels,
            flags=flags,
        )

        # assert
        # result with masked off nan values
        assert allclose(result[np.isfinite(result)], expected)
        # since array_data has a nan
        # when highpass flag is on
        # result should also have a nan at the same position
        if FilterFlags.HIGH_PASS in flags:
            assert np.isnan(result).any()

    def test_both_flag(
        self,
        alpha: float,
        array_data: ScanMap2DArray,
        cutoff_pixels: NDArray[np.floating],
    ) -> None:
        # act
        result = apply_gaussian_filter(
            array_data,
            alpha,
            cutoff_pixels=cutoff_pixels,
            flags=FilterFlags.HIGH_PASS & FilterFlags.NAN_OUT,
        )

        # assert
        assert allclose(result, 5.0)

    def test_compaire_high_against_low_pass(
        self,
        alpha: float,
        array_data: ScanMap2DArray,
        cutoff_pixels: NDArray[np.floating],
    ) -> None:
        # arrange, filter acts different if nan in data
        array = array_data.copy()
        array[5, 5] = 5.0
        filter = partial(
            apply_gaussian_filter,
            data=array,
            alpha=alpha,
            cutoff_pixels=cutoff_pixels,
        )

        # act
        lowpass = filter(flags=~(FilterFlags.HIGH_PASS | FilterFlags.NAN_OUT))
        highpass = filter(flags=FilterFlags.HIGH_PASS & ~FilterFlags.NAN_OUT)

        # assert
        assert allclose(lowpass + highpass, array)

    def test_single_pixel(
        self, alpha: float, cutoff_pixels: NDArray[np.floating]
    ) -> None:
        """Single pixel data should work."""
        data = np.array([[42.0]])
        result = apply_gaussian_filter(data, alpha, cutoff_pixels=cutoff_pixels)

        assert result.shape == (1, 1)
        assert result[0, 0] == approx(42.0)

    def test_all_nan_data(
        self, alpha: float, cutoff_pixels: NDArray[np.floating]
    ) -> None:
        """All-NaN data should return all NaN."""
        # arrange
        data = np.full((10, 10), np.nan)
        # act
        result = apply_gaussian_filter(data, alpha, cutoff_pixels=cutoff_pixels)
        # assert
        assert np.isnan(result).all()

    def test_single_valid_value(
        self, alpha: float, cutoff_pixels: NDArray[np.floating]
    ) -> None:
        """Single non-NaN value should be preserved."""
        # arrange
        data = np.full((10, 10), np.nan)
        data[5, 5] = 5.0

        # act
        result = apply_gaussian_filter(data, alpha, cutoff_pixels=cutoff_pixels)

        # assert
        assert result[5, 5] == approx(5.0)

    def test_row_of_nan(
        self, alpha: float, cutoff_pixels: NDArray[np.floating]
    ) -> None:
        """Row of NaN should be handled correctly."""
        # arrange
        data = np.random.rand(20, 20) * 100
        data[10, :] = np.nan

        # act
        result = apply_gaussian_filter(data, alpha, cutoff_pixels=cutoff_pixels)

        # assert
        assert np.isnan(result[10, :]).all()
        assert not np.isnan(result[5, :]).all()

    def test_column_of_nan(
        self, alpha: float, cutoff_pixels: NDArray[np.floating]
    ) -> None:
        """Column of NaN should be handled correctly."""
        # arrange
        data = np.random.rand(20, 20) * 100
        data[:, 10] = np.nan

        # act
        result = apply_gaussian_filter(data, alpha, cutoff_pixels=cutoff_pixels)

        # assert
        assert np.isnan(result[:, 10]).all()
        assert not np.isnan(result[:, 5]).all()

    def test_nan_border(
        self, alpha: float, cutoff_pixels: NDArray[np.floating]
    ) -> None:
        """NaN border should be preserved."""
        # arrange
        data = np.full((20, 20), np.nan)
        data[5:15, 5:15] = np.random.rand(10, 10) * 100

        # act
        result = apply_gaussian_filter(data, alpha, cutoff_pixels=cutoff_pixels)

        # assert
        assert np.isnan(result[:5, :]).all()
        assert np.isnan(result[15:, :]).all()
        assert np.isnan(result[:, :5]).all()
        assert np.isnan(result[:, 15:]).all()

    def test_very_small_cutoff(self, alpha: float) -> None:
        """Very small cutoff should approximate identity."""
        # arrange
        data = np.random.rand(20, 20) * 100

        # act
        result = apply_gaussian_filter(data, alpha, cutoff_pixels=np.array([0.1, 0.1]))

        # assert
        assert allclose(result[5:15, 5:15], data[5:15, 5:15])

    def test_very_large_cutoff(self, alpha: float) -> None:
        """Very large cutoff should approximate mean."""
        # arrange
        data = np.random.rand(20, 20) * 100

        # act
        result = apply_gaussian_filter(
            data, alpha, cutoff_pixels=np.array([100.0, 100.0])
        )
        interior = result[8:12, 8:12]

        # assert
        assert np.mean(interior) == pytest.approx(np.mean(data), rel=deci)
        assert np.std(interior) < 1.0
