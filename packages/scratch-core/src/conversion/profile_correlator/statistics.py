"""
Statistical metrics for profile comparison.

This module provides functions for computing statistical metrics between
1D profiles, including:

- compute_cross_correlation: NaN-aware normalized cross-correlation
- compute_roughness_sa: Arithmetic mean roughness (Sa)
- compute_roughness_sq: Root mean square roughness (Sq)
- compute_overlap_ratio: Overlap ratio relative to shorter profile
- compute_signature_differences: Normalized signature difference metrics

All length and height measurements are in meters (SI units).
"""

import numpy as np
from numpy.typing import NDArray


def compute_cross_correlation(
    profile_1: NDArray[np.floating],
    profile_2: NDArray[np.floating],
) -> float:
    """
    Compute normalized cross-correlation between two profiles.

    This function computes the Pearson correlation coefficient between two
    1D profiles, properly handling NaN values by excluding them from the
    calculation. Both profiles are mean-centered before computing the
    correlation.

    The formula used is::

        r = (p1' * p2) / sqrt((p1' * p1) * (p2' * p2))

    where p1 and p2 are the mean-centered profiles with NaN values removed.

    :param profile_1: First profile as a 1D array. May contain NaN values.
    :param profile_2: Second profile as a 1D array. Must have the same length
        as profile_1. May contain NaN values.
    :returns: Correlation coefficient in the range [-1, 1]. Returns NaN if
        there are no valid (non-NaN) overlapping samples.
    :raises ValueError: If profiles have different lengths.
    """
    # Ensure 1D arrays
    profile_1 = np.asarray(profile_1).ravel()
    profile_2 = np.asarray(profile_2).ravel()

    # Validate lengths
    if len(profile_1) != len(profile_2):
        raise ValueError(
            f"Profiles must have the same length. "
            f"Got {len(profile_1)} and {len(profile_2)}."
        )

    # Find indices where both profiles have valid (non-NaN) values
    valid_mask = ~(np.isnan(profile_1) | np.isnan(profile_2))

    # Extract valid samples
    p1_valid = profile_1[valid_mask]
    p2_valid = profile_2[valid_mask]

    # Check if we have any valid samples
    n_valid = len(p1_valid)
    if n_valid == 0:
        return np.nan

    # Subtract the mean from the profile. (Results in profiles with 0 mean)
    p1_centered = p1_valid - np.mean(p1_valid)
    p2_centered = p2_valid - np.mean(p2_valid)

    # Compute correlation terms
    a12 = np.dot(p1_centered, p2_centered)
    a11 = np.dot(p1_centered, p1_centered)
    a22 = np.dot(p2_centered, p2_centered)

    # Compute correlation coefficient
    denominator = np.sqrt(a11 * a22)
    if denominator == 0:
        # Both profiles are constant (zero variance)
        return np.nan

    return float(a12 / denominator)


def compute_roughness_sa(profile: NDArray[np.floating]) -> float:
    """
    Compute arithmetic mean roughness (Sa) of a profile.

    Sa is the arithmetic mean of the absolute values of the profile heights,
    calculated as: Sa = mean(|z|)

    :param profile: 1D profile array. May contain NaN values which are ignored.
    :returns: Sa value in the same units as the input profile.
    """
    return float(np.nanmean(np.abs(profile)))


def compute_roughness_sq(profile: NDArray[np.floating]) -> float:
    """
    Compute root mean square roughness (Sq) of a profile.

    Sq is the root mean square of the profile heights, calculated as:
    Sq = sqrt(mean(z^2))

    :param profile: 1D profile array. May contain NaN values which are ignored.
    :returns: Sq value in the same units as the input profile.
    """
    return float(np.sqrt(np.nanmean(profile**2)))


def compute_overlap_ratio(
    overlap_length: float,
    ref_length: float,
    comp_length: float,
) -> float:
    """
    Compute overlap ratio relative to the shorter profile.

    The overlap ratio indicates what fraction of the shorter profile is
    covered by the overlap region after alignment.

    :param overlap_length: Length of the overlap region (in meters).
    :param ref_length: Length of the reference profile (in meters).
    :param comp_length: Length of the comparison profile (in meters).
    :returns: Overlap ratio in range [0, 1]. Returns NaN if shorter length is 0.
    """
    shorter_length = min(ref_length, comp_length)
    if shorter_length == 0:
        return np.nan
    return overlap_length / shorter_length


def compute_signature_differences(
    sq_diff: float,
    sq_ref: float,
    sq_comp: float,
) -> tuple[float, float, float]:
    """
    Compute normalized signature difference metrics.

    These metrics quantify the difference between profiles normalized by
    their roughness, providing dimensionless measures of dissimilarity.

    :param sq_diff: Sq of the difference profile (comp - ref).
    :param sq_ref: Sq of the reference profile.
    :param sq_comp: Sq of the comparison profile.
    :returns: Tuple of (ds_ref_norm, ds_comp_norm, ds_combined) where:
        - ds_ref_norm: (Sq_diff / Sq_ref)^2 - normalized to reference
        - ds_comp_norm: (Sq_diff / Sq_comp)^2 - normalized to comparison
        - ds_combined: Sq_diff^2 / (Sq_ref * Sq_comp) - geometric mean normalization
        Returns NaN for any metric where division by zero would occur.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ds_ref_norm = (sq_diff / sq_ref) ** 2 if sq_ref != 0 else np.nan
        ds_comp_norm = (sq_diff / sq_comp) ** 2 if sq_comp != 0 else np.nan
        ds_combined = (
            sq_diff**2 / (sq_ref * sq_comp)
            if (sq_ref != 0 and sq_comp != 0)
            else np.nan
        )
    return ds_ref_norm, ds_comp_norm, ds_combined
