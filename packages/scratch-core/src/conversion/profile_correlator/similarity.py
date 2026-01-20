"""Similarity metrics for profile comparison.

This module provides functions for computing similarity metrics between
1D profiles, including cross-correlation and comprehensive comparison
statistics for striated mark analysis.

The main functions are:
- compute_cross_correlation: NaN-aware normalized cross-correlation
- compute_comparison_metrics: Full set of comparison metrics

These correspond to the MATLAB functions:
- GetSimilarityScore.m
- GetStriatedMarkComparisonResults.m
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

    This corresponds to MATLAB's GetSimilarityScore.m with score_type='cross_correlation'.

    :param profile_1: First profile as a 1D array. May contain NaN values.
    :param profile_2: Second profile as a 1D array. Must have the same length
        as profile_1. May contain NaN values.
    :returns: Correlation coefficient in the range [-1, 1]. Returns NaN if
        there are no valid (non-NaN) overlapping samples.
    :raises ValueError: If profiles have different lengths.

    Example::

        >>> import numpy as np
        >>> p1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> p2 = np.array([1.1, 2.1, 2.9, 4.0, 5.1])
        >>> r = compute_cross_correlation(p1, p2)
        >>> r > 0.99
        True
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
    # This corresponds to MATLAB: ind = isnan(profile_1)|isnan(profile_2)
    valid_mask = ~(np.isnan(profile_1) | np.isnan(profile_2))

    # Extract valid samples
    p1_valid = profile_1[valid_mask]
    p2_valid = profile_2[valid_mask]

    # Check if we have any valid samples
    n_valid = len(p1_valid)
    if n_valid == 0:
        return np.nan

    # Mean-center the profiles
    # MATLAB: profile_1 = profile_1 - sum(profile_1,1)/size(profile_1,1)
    p1_centered = p1_valid - np.mean(p1_valid)
    p2_centered = p2_valid - np.mean(p2_valid)

    # Compute correlation terms
    # MATLAB: a12 = profile_1' * profile_2
    a12 = np.dot(p1_centered, p2_centered)
    a11 = np.dot(p1_centered, p1_centered)
    a22 = np.dot(p2_centered, p2_centered)

    # Compute correlation coefficient
    # MATLAB: similarity_score = a12/sqrt(a11*a22)
    denominator = np.sqrt(a11 * a22)
    if denominator == 0:
        # Both profiles are constant (zero variance)
        return np.nan

    return float(a12 / denominator)


def compute_comparison_metrics(
    transforms: Sequence[TransformParameters],
    profile_ref: NDArray[np.floating],
    profile_comp: NDArray[np.floating],
    pixel_size_um: float,
) -> ComparisonResults:
    """
    Compute complete set of comparison metrics for striated marks.

    This function calculates all metrics needed for forensic comparison of
    striated marks, including registration parameters, roughness measurements,
    and signature differences. The metrics are computed after alignment.

    The metrics computed are:
    - Registration: position shift, scale factor
    - Roughness: Sa (mean absolute height), Sq (RMS roughness) for each profile
    - Difference: Sa12, Sq12 for the difference between profiles
    - Signature differences: ds1, ds2, ds (various normalizations)

    This corresponds to MATLAB's GetStriatedMarkComparisonResults.m.

    :param transforms: Sequence of TransformParameters from the alignment,
        one per scale level. Used to compute cumulative transformation.
    :param profile_ref: Reference profile (aligned) in meters.
    :param profile_comp: Compared profile (aligned) in meters.
    :param pixel_size_um: Pixel separation in micrometers.
    :returns: ComparisonResults with all metrics populated.

    Example::

        >>> transforms = [TransformParameters(translation=5.0, scaling=1.001)]
        >>> ref = np.random.randn(100) * 1e-6  # Profile in meters
        >>> comp = np.random.randn(100) * 1e-6
        >>> results = compute_comparison_metrics(transforms, ref, comp, 0.5)
        >>> results.correlation_coefficient  # Will be some value
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
    # MATLAB code builds transform_matrix by multiplying new * old
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

    # Compute position shift in micrometers
    # MATLAB: results_table.dPos = transform_matrix(1,3)*results_table.vPixSep1
    position_shift = total_translation * pixel_size_um

    # Compute correlation coefficient
    correlation = compute_cross_correlation(profile_ref, profile_comp)

    # Compute overlap length
    # MATLAB: results_table.lOverlap = length(profiles1)*results_table.vPixSep1
    n_samples = len(profile_ref)
    overlap_length = n_samples * pixel_size_um

    # Convert profiles to micrometers for roughness calculations
    # MATLAB: profiles1 = profiles1.*1e6
    p1_um = profile_ref * 1e6
    p2_um = profile_comp * 1e6
    p_diff_um = p2_um - p1_um

    # Compute roughness parameters
    # Sa = mean absolute height: sum(abs(profile))/N
    # Sq = RMS roughness: sqrt((profile' * profile)/N)
    #
    # MATLAB:
    # results_table.sa_1 = sum(abs(profiles1),1)/N;
    # results_table.sq_1 = sqrt((profiles1'*profiles1)/N);
    sa_ref = float(np.mean(np.abs(p1_um)))
    sq_ref = float(np.sqrt(np.mean(p1_um**2)))

    sa_comp = float(np.mean(np.abs(p2_um)))
    sq_comp = float(np.sqrt(np.mean(p2_um**2)))

    sa_diff = float(np.mean(np.abs(p_diff_um)))
    sq_diff = float(np.sqrt(np.mean(p_diff_um**2)))

    # Compute signature differences
    # MATLAB:
    # results_table.ds1 = (results_table.sq12/results_table.sq_1)^2;
    # results_table.ds2 = (results_table.sq12/results_table.sq_2)^2;
    # results_table.ds  = results_table.sq12^2/(results_table.sq_1*results_table.sq_2);
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
        pixel_size_ref=pixel_size_um,
        pixel_size_comp=pixel_size_um,
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
