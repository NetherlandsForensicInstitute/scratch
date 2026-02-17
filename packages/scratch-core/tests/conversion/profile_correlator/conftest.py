"""
Shared test fixtures for profile_correlator tests.

This module provides fixtures for creating synthetic profiles with known
properties, useful for testing the profile correlation functions.
"""

from pathlib import Path

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.profile_correlator import Profile, AlignmentParameters


# Directory for test data files (MATLAB .mat files for validation)
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def pixel_size_05um() -> float:
    """Standard pixel size of 0.5 micrometers in meters."""
    return 0.5e-6


@pytest.fixture
def pixel_size_1um() -> float:
    """Standard pixel size of 1.0 micrometer in meters."""
    return 1.0e-6


@pytest.fixture
def simple_sine_profile(pixel_size_05um: float) -> Profile:
    """
    Create a simple sinusoidal profile for basic tests.

    The profile contains 1000 samples of a sine wave with some added noise.
    """
    np.random.seed(42)
    x = np.linspace(0, 10 * np.pi, 1000)
    data = np.sin(x) * 1e-6  # Heights in micrometers scale
    data += np.random.normal(0, 0.01e-6, len(data))  # Add a small noise

    return Profile(heights=data, pixel_size=pixel_size_05um)


@pytest.fixture
def shifted_sine_profile(pixel_size_05um: float) -> Profile:
    """
    Create a shifted version of the sine profile for alignment tests.

    The profile is shifted by approximately 20 samples.
    """
    np.random.seed(43)  # Different seed for different noise
    x = np.linspace(0, 10 * np.pi, 1000)
    shift = 0.2  # radians, approximately 20 samples
    data = np.sin(x + shift) * 1e-6
    data += np.random.normal(0, 0.01e-6, len(data))

    return Profile(heights=data, pixel_size=pixel_size_05um)


@pytest.fixture
def scaled_sine_profile(pixel_size_05um: float) -> Profile:
    """
    Create a scaled (stretched) version of the sine profile.

    The profile is scaled by 1.02 (2% stretch).
    """
    np.random.seed(44)
    # Create profile with 1.02x scale (fewer periods in same length)
    x = np.linspace(0, 10 * np.pi / 1.02, 1000)
    data = np.sin(x) * 1e-6
    data += np.random.normal(0, 0.01e-6, len(data))

    return Profile(heights=data, pixel_size=pixel_size_05um)


@pytest.fixture
def profile_with_nans(pixel_size_05um: float) -> Profile:
    """Create a profile with some NaN values for NaN handling tests."""
    np.random.seed(45)
    x = np.linspace(0, 10 * np.pi, 1000)
    data = np.sin(x) * 1e-6
    data += np.random.normal(0, 0.01e-6, len(data))

    # Insert some NaN values
    data[100:110] = np.nan  # Block of NaNs
    data[500] = np.nan  # Single NaN
    data[700:750] = np.nan  # Larger block

    return Profile(heights=data, pixel_size=pixel_size_05um)


@pytest.fixture
def partial_profile(pixel_size_05um: float) -> Profile:
    """
    Create a partial (shorter) profile for partial matching tests.

    This profile is a subset of a longer reference, starting at index 300
    and having length 400.
    """
    np.random.seed(46)
    x = np.linspace(0, 10 * np.pi, 1000)
    full_data = np.sin(x) * 1e-6

    # Extract partial segment
    partial_data = full_data[300:700].copy()
    partial_data += np.random.normal(0, 0.01e-6, len(partial_data))

    return Profile(heights=partial_data, pixel_size=pixel_size_05um)


@pytest.fixture
def different_resolution_profile(pixel_size_1um: float) -> Profile:
    """Create a profile with different pixel size for resampling tests."""
    np.random.seed(48)
    # Half the number of samples due to double pixel size
    x = np.linspace(0, 10 * np.pi, 500)
    data = np.sin(x) * 1e-6
    data += np.random.normal(0, 0.01e-6, len(data))

    return Profile(heights=data, pixel_size=pixel_size_1um)


@pytest.fixture
def default_params() -> AlignmentParameters:
    """Default alignment parameters for tests."""
    return AlignmentParameters()


@pytest.fixture
def fast_params() -> AlignmentParameters:
    """Fast alignment parameters for quicker tests."""
    return AlignmentParameters(
        max_scaling=0.05,
    )


def make_synthetic_striation_profile(
    n_samples: int = 1000,
    n_striations: int = 20,
    amplitude_um: float = 0.5,
    noise_level: float = 0.05,
    pixel_size_m: float = 0.5e-6,
    seed: int | None = None,
) -> Profile:
    """
    Create a synthetic striation profile for testing.

    This function generates a profile that mimics the appearance of striation
    marks with multiple ridges and valleys.

    :param n_samples: Number of samples in the profile.
    :param n_striations: Number of striation features.
    :param amplitude_um: Amplitude of striations in micrometers.
    :param noise_level: Relative noise level (0 to 1).
    :param pixel_size_m: Pixel size in meters.
    :param seed: Random seed for reproducibility.
    :returns: Profile with synthetic striation data.
    """
    if seed is not None:
        np.random.seed(seed)

    # Create base profile with multiple frequency components
    x = np.linspace(0, n_striations * 2 * np.pi, n_samples)

    # Primary striation pattern
    data = np.sin(x) * amplitude_um * 1e-6

    # Add some harmonics for realism
    data += np.sin(2 * x) * amplitude_um * 0.3 * 1e-6
    data += np.sin(0.5 * x) * amplitude_um * 0.5 * 1e-6

    # Add noise
    noise = np.random.normal(0, amplitude_um * noise_level * 1e-6, n_samples)
    data += noise

    return Profile(heights=data, pixel_size=pixel_size_m)


def striation_mark(profile: Profile, n_cols: int = 50) -> Mark:
    """
    Build a 2D striation Mark by tiling a profile across columns.

    :param profile: Source profile whose heights become the row data.
    :param n_cols: Number of columns in the resulting mark.
    :returns: Mark with data shape (len(profile.heights), n_cols).
    """
    data = np.tile(profile.heights[:, np.newaxis], (1, n_cols))
    return Mark(
        scan_image=ScanImage(
            data=data,
            scale_x=profile.pixel_size,
            scale_y=profile.pixel_size,
        ),
        mark_type=MarkType.BULLET_GEA_STRIATION,
    )


def make_shifted_profile(
    profile: Profile,
    shift_samples: float,
    scale_factor: float = 1.0,
    seed: int | None = None,
) -> Profile:
    """
    Create a shifted and optionally scaled version of a profile.

    :param profile: Original profile.
    :param shift_samples: Number of samples to shift (can be fractional).
    :param scale_factor: Scaling factor (1.0 = no scaling).
    :param seed: Random seed for added noise.
    :returns: New Profile with shifted/scaled data.
    """
    from scipy.interpolate import interp1d

    if seed is not None:
        np.random.seed(seed)

    data = profile.heights
    n = len(data)

    # Create interpolator
    x_orig = np.arange(n)
    interpolator = interp1d(
        x_orig, data, kind="linear", fill_value=0, bounds_error=False
    )

    # Create new coordinates with shift and scale
    x_new = x_orig * scale_factor + shift_samples
    new_data = interpolator(x_new)

    # Add a small amount of noise
    new_data += np.random.normal(0, np.nanstd(data) * 0.01, n)

    return Profile(heights=new_data, pixel_size=profile.pixel_size)
