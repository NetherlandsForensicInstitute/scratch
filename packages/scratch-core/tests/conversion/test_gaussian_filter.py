"""Tests for Gaussian filter."""

from functools import partial

import numpy as np
import pytest

from conversion.gaussian_filter import (
    apply_gaussian_filter,
    get_alpha,
    get_sigmas,
)

CUTOFF: float = 5.0
SEED: int = 42
approx = partial(pytest.approx, rel=1e-15)
allclose = partial(np.allclose, rtol=1e-15)


@pytest.fixture(scope="module")
def alpha() -> float:
    return get_alpha(regression_order=0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


class TestGetAlpha:
    def test_order_0_and_1_return_same_value(self):
        assert get_alpha(0) == get_alpha(1)

    def test_order_2_returns_different_value(self):
        assert get_alpha(2) != get_alpha(0)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="Maximum regression order is 2"):
            get_alpha(3)


class TestCutoffToSigma:
    """Test cutoff to sigma conversion."""

    def test_sigma_calculated_from_cutoffs(self, alpha: float):
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
        sigmas = get_sigmas(alpha, cutoff_lengths=cutoffs)

        # assert
        assert sigmas == approx(expected)
        assert (sigmas < cutoffs).all()

    def test_sigma_scales_linearly(self, alpha: float):
        # arrange
        cutoffs = np.array([2, 4, 8, 16, 32])

        # act and assert
        sigmas = get_sigmas(alpha, cutoff_lengths=cutoffs)
        assert sigmas[1:] == approx(sigmas[:-1] * 2)

    def test_zero_cutoffs_gives_zero_sigma(self, alpha: float) -> None:
        # arrange
        cutoffs = np.array([0])

        # act
        sigmas = get_sigmas(alpha, cutoff_lengths=cutoffs)

        # assert
        assert np.array_equal(cutoffs, sigmas)


class TestGaussianFilterFunction:
    """Test the gaussian_filter function."""

    def test_nan_out_false(self):
        """When nan_out=False, NaN positions get interpolated values."""
        data = np.ones((10, 10)) * 5.0
        data[5, 5] = np.nan

        result = apply_gaussian_filter(data, (3.0, 3.0), nan_out=False)

        # The NaN position should now have a value (interpolated from neighbors)
        assert not np.isnan(result[5, 5])
        assert result[5, 5] == pytest.approx(5.0, rel=0.1)

    def test_nan_out_true(self):
        """When nan_out=True, NaN positions remain NaN."""
        data = np.ones((10, 10)) * 5.0
        data[5, 5] = np.nan

        result = apply_gaussian_filter(data, (3.0, 3.0), nan_out=True)

        # The NaN position should now have a value (interpolated from neighbors)
        assert np.isnan(result[5, 5])

    def test_asymmetric_pixel_separation(self, rng: np.random.Generator):
        """Should handle different pixel separation in each direction."""
        data = rng.random((30, 30)) * 100

        # Different separation in row vs col
        result = apply_gaussian_filter(data, (5.0, 5.0), pixel_size=(1.0, 0.5))

        assert result.shape == data.shape
        assert not np.any(np.isnan(result))

    def test_preserves_shape(self, rng: np.random.Generator):
        """Output should have same shape as input."""
        for shape in [(10, 10), (20, 30), (50, 25)]:
            data = rng.random(shape)
            result = apply_gaussian_filter(data, (5.0, 5.0))
            assert result.shape == shape

    def test_uniform_data_unchanged(self):
        """Filtering uniform data should return (approximately) same values."""
        data = np.ones((20, 20)) * 42.0
        result = apply_gaussian_filter(data, (5.0, 5.0))

        # Interior should be exactly 42 (edges affected by zero-padding)
        np.testing.assert_allclose(result[5:15, 5:15], 42.0, rtol=1e-10)

    def test_smoothing_effect(self, rng: np.random.Generator):
        """Filtering should reduce variance (smoothing)."""
        data = rng.random((50, 50)) * 100
        result = apply_gaussian_filter(data, (5.0, 5.0))

        # Variance should decrease after smoothing
        assert np.var(result) < np.var(data)

    def test_high_pass_equals_original_minus_low_pass(self, rng: np.random.Generator):
        """High-pass should return data - lowpass."""
        data = rng.random((30, 30)) * 100

        lowpass = apply_gaussian_filter(
            data, cutoff_lengths=(5.0, 5.0), is_high_pass=False
        )
        highpass = apply_gaussian_filter(
            data, cutoff_lengths=(5.0, 5.0), is_high_pass=True
        )

        np.testing.assert_allclose(highpass, data - lowpass, rtol=1e-15)

    def test_nan_cutoff_returns_unchanged(self, rng: np.random.Generator):
        """NaN cutoff should return data unchanged."""
        data = rng.random((10, 10))
        result = apply_gaussian_filter(
            data, cutoff_lengths=(np.nan, np.nan), is_high_pass=False
        )

        np.testing.assert_array_equal(result, data)

    def test_single_pixel_data_gives_single_pixel_result(self):
        data = np.array([[42.0]])
        result = apply_gaussian_filter(data, (3.0, 3.0))

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(42.0)

    def test_all_nan_data_returns_all_nan_result(self):
        data = np.full((10, 10), np.nan)
        result = apply_gaussian_filter(data, (5.0, 5.0))

        assert np.all(np.isnan(result))

    def test_single_valid_value_is_preserved(self):
        data = np.full((10, 10), np.nan)
        data[5, 5] = 42.0

        result = apply_gaussian_filter(data, (3.0, 3.0))

        assert result[5, 5] == pytest.approx(42.0)

    def test_row_of_nan_is_preserved(self, rng: np.random.Generator):
        data = rng.random((20, 20)) * 100
        data[10, :] = np.nan

        result = apply_gaussian_filter(data, (5.0, 5.0))

        # NaN row should remain NaN
        assert np.all(np.isnan(result[10, :]))
        # Other rows should have values
        assert not np.all(np.isnan(result[5, :]))

    def test_column_of_nan_is_preserved(self, rng: np.random.Generator):
        data = rng.random((20, 20)) * 100
        data[:, 10] = np.nan

        result = apply_gaussian_filter(data, (5.0, 5.0))

        # NaN column should remain NaN
        assert np.all(np.isnan(result[:, 10]))
        # Other columns should have values
        assert not np.all(np.isnan(result[:, 5]))

    def test_nan_border_is_preserved(self, rng: np.random.Generator):
        data = np.full((20, 20), np.nan)
        data[5:15, 5:15] = rng.random((10, 10)) * 100

        result = apply_gaussian_filter(data, (3.0, 3.0))

        # NaN border should remain NaN
        assert np.all(np.isnan(result[:5, :]))
        assert np.all(np.isnan(result[15:, :]))
        assert np.all(np.isnan(result[:, :5]))
        assert np.all(np.isnan(result[:, 15:]))

    def test_very_small_cutoff_preserves_interior_array(self, rng: np.random.Generator):
        data = rng.random((20, 20)) * 100

        result = apply_gaussian_filter(data, (0.1, 0.1))

        # With very small cutoff, result should be close to original
        # (at least in the interior, away from edges)
        np.testing.assert_allclose(result[5:15, 5:15], data[5:15, 5:15], rtol=0.01)

    def test_very_large_cutoff_preserves_mean(self, rng: np.random.Generator):
        data = rng.random((20, 20)) * 100

        result = apply_gaussian_filter(data, (100.0, 100.0))

        # With very large cutoff, interior values should approach global mean
        # (edges are affected by zero-padding so we check interior)
        interior = result[8:12, 8:12]

        # Interior mean should be close to data mean (within reason, due to edge effects)
        assert np.mean(interior) == pytest.approx(np.mean(data), rel=0.5)

        # All interior values should be close to each other
        assert np.std(interior) < 1.0

    def test_output_in_input_range(self, rng: np.random.Generator):
        """Filter output should be within input range (for non-edge pixels)."""
        data = rng.random((50, 50)) * 100
        result = apply_gaussian_filter(data, (5.0, 5.0))

        # Interior values should be within input range
        interior = result[10:40, 10:40]
        assert np.min(interior) >= np.min(data) - 1e-10
        assert np.max(interior) <= np.max(data) + 1e-10
