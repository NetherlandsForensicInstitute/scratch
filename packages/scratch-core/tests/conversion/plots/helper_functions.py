"""
Helper functions for testing plot functions.

Includes helpers for:
- Validating outputs (RGB images)
- Creating synthetic test data (impressions, striations, profiles)
- Creating sample metadata and score data for CCF comparison tests
"""

import numpy as np
from container_models.base import FloatArray1D, FloatArray2D


def assert_valid_rgb_image(arr: np.ndarray) -> None:
    """
    Assert that an array is a valid RGB image.
    
    :param arr: Array to check
    :raises AssertionError: If array is not a valid RGB uint8 image
    """
    assert arr.ndim == 3, f"Expected 3D array, got {arr.ndim}D"
    assert arr.shape[2] == 3, f"Expected 3 channels (RGB), got {arr.shape[2]}"
    assert arr.dtype == np.uint8, f"Expected uint8 dtype, got {arr.dtype}"
    assert arr.size > 0, "Image array is empty"


def create_synthetic_impression_data(
    height: int, width: int, seed: int = 42
) -> FloatArray2D:
    """
    Create synthetic impression/depth map data for testing.
    
    This creates data that simulates ballistic impression marks
    with more random, less structured patterns compared to striations.
    
    :param height: Height of the impression in pixels
    :param width: Width of the impression in pixels
    :param seed: Random seed for reproducibility
    :return: 2D array of synthetic impression data in meters
    
    Example::
    
        >>> data = create_synthetic_impression_data(200, 200, seed=42)
        >>> print(data.shape)
        (200, 200)
    """
    np.random.seed(seed)
    
    # Create random impression pattern
    impression = np.random.randn(height, width) * 0.3
    
    # Add some larger-scale features (like firing pin impressions)
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Central depression (like a firing pin mark)
    center_y, center_x = height // 2, width // 2
    r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    central_feature = -0.5 * np.exp(-(r**2) / (0.1 * min(height, width)**2))
    
    # Some radial patterns
    radial_pattern = 0.2 * np.sin(r / 10) * np.exp(-r / (0.3 * min(height, width)))
    
    # Combine all features
    impression = impression + central_feature + radial_pattern
    
    # Convert to meters
    return impression * 1e-6


def create_synthetic_impression_mark(
    height: int, width: int, seed: int = 42
) -> FloatArray2D:
    """
    Create synthetic impression mark data for testing.
    
    Alias for create_synthetic_impression_data(). Creates ballistic
    impression marks with random structure and features like firing
    pin impressions.
    
    :param height: Height of the impression in pixels
    :param width: Width of the impression in pixels
    :param seed: Random seed for reproducibility
    :return: 2D array of synthetic impression mark data in meters
    
    Example::
    
        >>> mark = create_synthetic_impression_mark(200, 200, seed=42)
        >>> print(mark.shape)
        (200, 200)
    """
    # Just call the main impression data function
    return create_synthetic_impression_data(height, width, seed)


def create_synthetic_striation_data(
    height: int, width: int, seed: int = 42
) -> FloatArray2D:
    """
    Create synthetic striation surface data for testing.
    
    :param height: Height of the surface in pixels
    :param width: Width of the surface in pixels
    :param seed: Random seed for reproducibility
    :return: 2D array of synthetic striation data in meters
    """
    np.random.seed(seed)
    
    # Create horizontal striations (variation along y-axis)
    y = np.arange(height).reshape(-1, 1)
    
    # Multiple frequency components
    surface = (
        0.5 * np.sin(y / 50) +
        0.3 * np.sin(y / 20) +
        0.2 * np.sin(y / 10) +
        0.1 * np.random.randn(height, width)
    ) * 1e-6  # Convert to meters
    
    return surface


def create_synthetic_profile_data(length: int, seed: int = 42) -> FloatArray2D:
    """
    Create synthetic 1D profile data for testing.
    
    :param length: Length of the profile
    :param seed: Random seed for reproducibility
    :return: 1D array (as 2D with height=1) of profile data in meters
    """
    np.random.seed(seed)
    
    x = np.arange(length)
    profile = (
        0.5 * np.sin(x / 30) +
        0.3 * np.cos(x / 15) +
        0.1 * np.random.randn(length)
    ) * 1e-6
    
    return profile.reshape(1, -1)


def create_sample_score_data(n_knm: int = 1000, n_km: int = 100, seed: int = 42) -> dict:
    """
    Create sample score distribution data for testing CCF comparison plots.
    
    :param n_knm: Number of known non-match scores
    :param n_km: Number of known match scores  
    :param seed: Random seed for reproducibility
    :return: Dictionary with scores, labels, and transformed data
    
    Example::
    
        >>> data = create_sample_score_data()
        >>> print(data.keys())
        dict_keys(['scores', 'labels', 'scores_transformed', 'llrs', 'llrs_at5', 'llrs_at95'])
    """
    np.random.seed(seed)
    
    # Create bimodal distributions
    knm_scores = np.random.beta(2, 5, n_knm)
    km_scores = np.random.beta(8, 2, n_km)
    
    scores = np.concatenate([knm_scores, km_scores])
    labels = np.concatenate([np.zeros(n_knm), np.ones(n_km)])
    
    # Shuffle
    idx = np.random.permutation(len(scores))
    scores = scores[idx]
    labels = labels[idx]
    
    # Transformed scores
    scores_transformed = 0.52 + scores * 0.47
    
    # LLR data
    n_llr = 100
    score_grid = np.linspace(
        scores_transformed.min(), scores_transformed.max(), n_llr
    )
    llrs = 5 * (score_grid - 0.75) ** 2 - 2
    llrs_at5 = llrs - 0.5
    llrs_at95 = llrs + 0.5
    
    return {
        "scores": scores,
        "labels": labels,
        "scores_transformed": scores_transformed,
        "llrs": llrs,
        "llrs_at5": llrs_at5,
        "llrs_at95": llrs_at95,
    }


def create_sample_metadata_reference() -> dict[str, str]:
    """
    Create sample reference metadata for testing.
    
    :return: Dictionary of metadata key-value pairs
    """
    return {
        "Case ID": "2022_07_21_126",
        "Firearm ID": "unknown_firearm_4",
        "Specimen ID": "aapl1214nl",
        "Measurement ID": "slaghoedje",
    }


def create_sample_metadata_compared() -> dict[str, str]:
    """
    Create sample compared metadata for testing.
    
    :return: Dictionary of metadata key-value pairs
    """
    return {
        "Case ID": "2022_07_21_126",
        "Firearm ID": "unknown_firearm_7",
        "Specimen ID": "aapl1217nl",
        "Measurement ID": "slaghoedje",
    }


def create_sample_metadata_results() -> dict[str, str]:
    """
    Create sample results metadata for testing.
    
    :return: Dictionary of results/metrics
    """
    return {
        "Date report": "2023-02-16",
        "User ID": "RUHES (apc_abal)",
        "Mark type": "Aperture shear striation",
        "Score type": "CCF",
        "Score (transform)": "0.97 (1.86)",
        "LogLR (5%, 95%)": "5.19 (5.17, 5.24)",
        "# of KM scores": "1144",
        "# of KNM scores": "296462",
    }

