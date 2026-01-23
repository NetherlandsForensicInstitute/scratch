"""
Minimal tests for preprocess_data.py and preprocess_data_filter.py.
One test per function.
"""

import numpy as np
import pytest

from conversion.preprocessing.preprocess_data_filter import (
    _apply_nan_weighted_gaussian_1d,
    _remove_zero_border,
    apply_gaussian_filter_1d,
)
from conversion.filter import ALPHA_GAUSSIAN
from conversion.preprocessing.preprocess_data import (
    cheby_cutoff_to_gauss_sigma as preprocess_cheby_cutoff_to_gauss_sigma,
    apply_shape_noise_removal,
)


# =============================================================================
# Tests for preprocess_data_filter.py
# =============================================================================


def test_cheby_cutoff_to_gauss_sigma():
    """Test conversion from cutoff wavelength to Gaussian sigma."""
    cutoff = 1000e-6  # 1000 µm
    pixel_size = 10e-6  # 10 µm per pixel

    sigma = preprocess_cheby_cutoff_to_gauss_sigma(cutoff, pixel_size)

    # sigma should be cutoff_pixels * ALPHA_GAUSSIAN
    expected = (cutoff / pixel_size) * ALPHA_GAUSSIAN
    assert sigma == pytest.approx(expected)


def test_apply_nan_weighted_gaussian_1d():
    """Test that NaN values are handled correctly in 1D Gaussian filter."""
    data = np.ones((20, 10), dtype=float)
    data[10, 5] = np.nan  # Single NaN value

    result = _apply_nan_weighted_gaussian_1d(data, sigma=2.0, radius=3)

    # Result should not have NaN propagation (NaN is interpolated)
    assert not np.isnan(result[10, 5])
    # Non-NaN regions should remain close to 1.0
    assert result[0, 0] == pytest.approx(1.0, rel=0.1)


def test_remove_zero_border():
    """Test that zero borders are correctly removed."""
    data = np.zeros((10, 10), dtype=float)
    mask = np.zeros((10, 10), dtype=bool)

    # Valid data only in center region
    data[3:7, 2:8] = 1.0
    mask[3:7, 2:8] = True

    cropped_data, cropped_mask, range_indices = _remove_zero_border(data, mask)

    assert cropped_data.shape == (4, 6)
    assert cropped_mask.shape == (4, 6)
    assert np.array_equal(range_indices, np.arange(3, 7))


def test_apply_gaussian_filter_1d_lowpass():
    """Test lowpass filtering removes high-frequency noise."""
    np.random.seed(42)

    # Create signal with low-frequency component + high-frequency noise
    rows = np.linspace(0, 10, 100)
    low_freq = np.sin(2 * np.pi * rows / 10)  # Low frequency signal
    high_freq = np.random.randn(100) * 0.5  # High frequency noise

    surface = np.tile((low_freq + high_freq).reshape(-1, 1), (1, 20))

    smoothed, indices, mask = apply_gaussian_filter_1d(
        surface,
        xdim=1e-6,
        cutoff=250e-6,
        is_high_pass=False,
        cut_borders_after_smoothing=False,
    )

    # Smoothed data should have lower variance than input
    assert np.std(smoothed) < np.std(surface)


def test_apply_gaussian_filter_1d_highpass():
    """Test highpass filtering removes low-frequency shape."""
    # Create surface with parabolic shape + fine detail
    # Use larger pixel size so cutoff captures the shape
    rows = np.linspace(0, 100, 200)
    shape = 0.01 * (rows - 50) ** 2  # Parabolic shape (range ~25)
    detail = np.sin(2 * np.pi * rows / 5) * 0.1  # Fine detail (range ~0.2)

    surface = np.tile((shape + detail).reshape(-1, 1), (1, 20))

    # Use pixel size that makes cutoff effective for shape removal
    # With xdim=1e-3, cutoff=50e-3 gives sigma ~47 pixels
    residuals, indices, mask = apply_gaussian_filter_1d(
        surface,
        xdim=1e-3,
        cutoff=50e-3,
        is_high_pass=True,
        cut_borders_after_smoothing=False,
    )

    # Residuals should have much smaller range than input (shape removed)
    # The parabolic shape (~25 range) should be mostly removed
    assert np.ptp(residuals) < np.ptp(surface) * 0.25


# =============================================================================
# Tests for preprocess_data.py
# =============================================================================


def test_apply_form_noise_removal():
    """Test complete form and noise removal pipeline."""
    np.random.seed(42)

    # Create synthetic surface: shape + striations + noise
    rows = np.linspace(0, 100, 300)
    shape = 0.001 * (rows - 50) ** 2  # Parabolic form
    striations = np.sin(2 * np.pi * rows / 10) * 0.01  # Striation pattern
    noise = np.random.randn(300) * 0.001  # Fine noise

    surface = np.tile((shape + striations + noise).reshape(-1, 1), (1, 50))

    result, mask = apply_shape_noise_removal(
        surface,
        xdim=1e-6,
        cutoff_hi=2000e-6,
        cutoff_lo=250e-6,
        cut_borders_after_smoothing=False,
    )

    # Result should have form removed (smaller range than input)
    assert np.ptp(result) < np.ptp(surface) * 0.3
    # Mask should be all True (no invalid regions)
    assert mask.all()
    # Output shape should match input when not cutting borders
    assert result.shape == surface.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
