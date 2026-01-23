"""
Similarity metrics for profile comparison.

This module provides functions for computing similarity metrics between
1D profiles, including cross-correlation and comprehensive comparison
statistics for striated mark analysis.

The main functions are:
- compute_cross_correlation: NaN-aware normalized cross-correlation
- compute_comparison_metrics: Full set of comparison metrics

All length and height measurements are in meters (SI units).
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from conversion.profile_correlator.data_types import (
    ComparisonResults,
    TransformParameters,
)


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

    # Mean-center the profiles
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


def compute_comparison_metrics(
    transforms: Sequence[TransformParameters],
    profile_ref: NDArray[np.floating],
    profile_comp: NDArray[np.floating],
    pixel_size: float,
) -> ComparisonResults:
    """
    Compute complete set of comparison metrics for striated marks.

    This function calculates all metrics needed for forensic comparison of
    striated marks, including registration parameters, roughness measurements,
    and signature differences. The metrics are computed after alignment.

    The metrics computed are:
    - Registration: position shift, scale factor
    - Roughness: Sa (mean absolute height), Sq (RMS roughness) for each profile
    - Difference: Sa_diff, Sq_diff for the difference between profiles
    - Signature differences: ds_ref_norm, ds_comp_norm, ds_combined

    All measurements are in meters (SI units).

    :param transforms: Sequence of TransformParameters from the alignment,
        one per scale level. Used to compute cumulative transformation.
    :param profile_ref: Reference profile (aligned) in meters.
    :param profile_comp: Compared profile (aligned) in meters.
    :param pixel_size: Pixel separation in meters.
    :returns: ComparisonResults with all metrics populated (in meters).
    """
    # Ensure 1D arrays
    profile_ref = np.asarray(profile_ref).ravel()
    profile_comp = np.asarray(profile_comp).ravel()

    # Compute cumulative transformation matrix
    # The transformation is: x' = scaling * x + translation
    # Composing multiple transforms: T_total = T_n * T_{n-1} * ... * T_1
    #
    # In matrix form (homogeneous coordinates):
    # [scaling  0  translation]   [s2  0  t2]   [s1  0  t1]
    # [   0     1      0      ] * [ 0  1   0] = [s2*s1  0  s2*t1+t2]
    # [   0     0      1      ]   [ 0  0   1]   [  0    0     1    ]
    #
    total_translation = 0.0
    total_scaling = 1.0

    for i, t in enumerate(transforms):
        if i == 0:
            total_translation = t.translation
            total_scaling = t.scaling
        else:
            # Compose: new transform applied after previous
            # x'' = s2 * (s1 * x + t1) + t2 = s2*s1*x + s2*t1 + t2
            total_translation = t.scaling * total_translation + t.translation
            total_scaling = t.scaling * total_scaling

    # Compute position shift in meters
    position_shift = total_translation * pixel_size

    # Compute correlation coefficient
    correlation = compute_cross_correlation(profile_ref, profile_comp)

    # Compute overlap length in meters
    n_samples = len(profile_ref)
    overlap_length = n_samples * pixel_size

    # Compute difference profile
    p_diff = profile_comp - profile_ref

    # Compute roughness parameters (all in meters)
    # Sa = mean absolute height: mean(|profile|)
    # Sq = RMS roughness: sqrt(mean(profile^2))
    sa_ref = float(np.mean(np.abs(profile_ref)))
    sq_ref = float(np.sqrt(np.mean(profile_ref**2)))

    sa_comp = float(np.mean(np.abs(profile_comp)))
    sq_comp = float(np.sqrt(np.mean(profile_comp**2)))

    sa_diff = float(np.mean(np.abs(p_diff)))
    sq_diff = float(np.sqrt(np.mean(p_diff**2)))

    # Compute signature differences (dimensionless ratios)
    with np.errstate(divide="ignore", invalid="ignore"):
        ds_ref_norm = (sq_diff / sq_ref) ** 2 if sq_ref != 0 else np.nan
        ds_comp_norm = (sq_diff / sq_comp) ** 2 if sq_comp != 0 else np.nan
        ds_combined = (
            sq_diff**2 / (sq_ref * sq_comp)
            if (sq_ref != 0 and sq_comp != 0)
            else np.nan
        )

    return ComparisonResults(
        is_profile_comparison=True,
        is_partial_profile=False,
        pixel_size_ref=pixel_size,
        pixel_size_comp=pixel_size,
        position_shift=position_shift,
        scale_factor=total_scaling,
        partial_start_position=np.nan,
        similarity_value=correlation,
        overlap_length=overlap_length,
        overlap_ratio=np.nan,  # Computed in correlator based on profile lengths
        correlation_coefficient=correlation,
        sa_ref=sa_ref,
        sq_ref=sq_ref,
        sa_comp=sa_comp,
        sq_comp=sq_comp,
        sa_diff=sa_diff,
        sq_diff=sq_diff,
        ds_ref_norm=ds_ref_norm,
        ds_comp_norm=ds_comp_norm,
        ds_combined=ds_combined,
    )
