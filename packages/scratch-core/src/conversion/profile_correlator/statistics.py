"""
Statistical metrics for profile comparison.

This module provides functions for computing statistical metrics between
1D profiles, including:

- compute_cross_correlation: NaN-aware normalized cross-correlation
- compute_roughness_sa: Arithmetic mean roughness
- compute_roughness_sq: Root mean square roughness
- compute_overlap_ratio: Overlap ratio relative to shorter profile
- compute_signature_differences: Normalized signature difference metrics

All length and height measurements are in meters (SI units).
"""

import numpy as np

from container_models.base import FloatArray1D
from conversion.profile_correlator.data_types import (
    RoughnessMetrics,
    SignatureDifferences,
)


def compute_cross_correlation(
    profile_1: FloatArray1D,
    profile_2: FloatArray1D,
) -> float | None:
    """
    Compute normalized cross-correlation between two profiles.

    This function computes the Pearson correlation coefficient between two
    1D profiles, properly handling NaN values by excluding them from the
    calculation.

    :param profile_1: First profile as a 1D array. May contain NaN values.
    :param profile_2: Second profile as a 1D array. Must have the same length
        as profile_1. May contain NaN values.
    :returns: Correlation coefficient in the range [-1, 1]. Returns NaN if
        there are fewer than 2 valid (non-NaN) overlapping samples or if
        either profile has zero variance.
    :raises ValueError: If profiles have different lengths.
    """
    profile_1 = np.asarray(profile_1).ravel()
    profile_2 = np.asarray(profile_2).ravel()

    if len(profile_1) != len(profile_2):
        raise ValueError(
            f"Profiles must have the same length. "
            f"Got {len(profile_1)} and {len(profile_2)}."
        )

    valid_mask = ~(np.isnan(profile_1) | np.isnan(profile_2))

    if np.sum(valid_mask) < 2:
        return None

    return float(np.corrcoef(profile_1[valid_mask], profile_2[valid_mask])[0, 1])


def compute_roughness_sa(profile: FloatArray1D) -> float:
    """
    Compute arithmetic mean roughness (ISO 25178 Sa parameter) of a profile.

    Sa is the arithmetic mean of the absolute values of the profile heights,
    calculated as: mean(|z|). The 'S' denotes a surface/areal parameter and
    'a' denotes arithmetical mean.

    :param profile: 1D profile array. May contain NaN values which are ignored.
    :returns: Arithmetic mean roughness (Sa) in the same units as the input profile.
    """
    return float(np.nanmean(np.abs(profile)))


def compute_roughness_sq(profile: FloatArray1D) -> float:
    """
    Compute root-mean-square roughness (ISO 25178 Sq parameter) of a profile.

    Sq is the root-mean-square of the profile heights, calculated as:
    sqrt(mean(z^2)). The 'S' denotes a surface/areal parameter and 'q'
    denotes quadratic mean (root-mean-square).

    :param profile: 1D profile array. May contain NaN values which are ignored.
    :returns: Root-mean-square roughness (Sq) in the same units as the input profile.
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

    :param overlap_length: Length of the overlap region in meters.
    :param ref_length: Length of the reference profile in meters.
    :param comp_length: Length of the comparison profile in meters.
    :returns: Overlap ratio in range [0, 1]. Returns NaN if shorter length is 0
        or if overlap_length exceeds shorter_length (invalid input).
    """
    shorter_length = min(ref_length, comp_length)
    if np.isclose(shorter_length, 0.0):
        return np.nan
    if overlap_length > shorter_length:
        return np.nan
    return overlap_length / shorter_length


def compute_signature_differences(roughness: RoughnessMetrics) -> SignatureDifferences:
    """
    Compute normalized signature difference metrics.

    These metrics quantify the difference between profiles normalized by
    their roughness, providing dimensionless measures of dissimilarity.

    :param roughness: Container with quadratic mean roughness (Sq) values for
        the reference profile, comparison profile, and difference profile.
    :returns: SignatureDifferences containing normalized metrics.
        Returns NaN for any metric where division by zero would occur.
    """
    sq_ref = roughness.sq_ref
    sq_comp = roughness.sq_comp
    sq_diff = roughness.sq_diff

    with np.errstate(divide="ignore", invalid="ignore"):
        ref_norm = (sq_diff / sq_ref) ** 2 if sq_ref > 0 else np.nan
        comp_norm = (sq_diff / sq_comp) ** 2 if sq_comp > 0 else np.nan
        combined = (
            sq_diff**2 / (sq_ref * sq_comp) if (sq_ref > 0 and sq_comp > 0) else np.nan
        )
    return SignatureDifferences(
        ref_norm=ref_norm, comp_norm=comp_norm, combined=combined
    )
