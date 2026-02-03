"""
Data types for profile correlation.

This module defines the core data structures used for profile correlation analysis
of striated marks. It follows the patterns established in preprocess_impression.

The main types are:
- Profile: Container for 1D profile data with metadata
- AlignmentParameters: Configuration for the brute-force alignment algorithm
- ComparisonResults: Full comparison metrics for striated mark analysis

All length and height measurements are in meters (SI units).
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Profile:
    """
    Container for a 1D profile with metadata.

    Profiles represent 1D height measurements along a scratch/striation mark.
    All measurements are in meters (SI units).

    :param depth_data: Height values as a 1D array of shape (N,).
    :param pixel_size: Physical distance between samples in meters.
    """

    depth_data: NDArray[np.floating]
    pixel_size: float

    @property
    def length(self) -> int:
        """
        Get the number of samples in the profile.

        :returns: Number of samples in depth_data.
        """
        return len(self.depth_data)


@dataclass(frozen=True)
class AlignmentParameters:
    """
    Configuration parameters for profile alignment.

    This dataclass contains parameters for the global brute-force alignment
    algorithm. The algorithm tries all possible shift positions and scale
    factors, selecting the combination with maximum cross-correlation.

    All length parameters are in meters (SI units).

    :param max_scaling: Maximum allowed scaling deviation as a fraction.
        E.g., 0.05 means scaling can vary from 0.95 to 1.05.
    :param min_overlap_distance: Minimum required overlap between profiles
        in meters. Alignments with less overlap are rejected.
    """

    max_scaling: float = 0.05
    min_overlap_distance: float = 350e-6  # 350 μm


@dataclass(frozen=True)
class ComparisonResults:
    """
    Full comparison metrics for striated mark analysis.

    This immutable dataclass contains all metrics computed during profile comparison,
    including registration parameters, roughness measurements, and
    signature differences.

    All length and height measurements are in meters (SI units).

    :param is_profile_comparison: True for full profile comparison mode.
    :param pixel_size_ref: Pixel separation of reference profile (m).
    :param pixel_size_comp: Pixel separation of compared profile (m).
    :param position_shift: Registration shift of compared profile
        relative to reference (m).
    :param scale_factor: Registration scale factor applied to compared
        profile (1.0 = no scaling).
    :param similarity_value: Optimized value of the similarity metric
        used during registration.
    :param overlap_length: Length of the overlapping region after
        registration (m).
    :param overlap_ratio: Ratio of overlap length to the length of the
        shorter profile.
    :param correlation_coefficient: Pearson cross-correlation coefficient
        between the aligned profiles.
    :param sa_ref: Mean absolute height (Sa) of the reference profile (m).
    :param sq_ref: RMS roughness (Sq) of the reference profile (m).
    :param sa_comp: Mean absolute height (Sa) of the compared profile (m).
    :param sq_comp: RMS roughness (Sq) of the compared profile (m).
    :param sa_diff: Mean absolute height difference between profiles (m).
    :param sq_diff: RMS height difference between profiles (m).
    :param ds_ref_norm: Signature difference normalized by reference Sq.
        Computed as (sq_diff / sq_ref)^2.
    :param ds_comp_norm: Signature difference normalized by compared Sq.
        Computed as (sq_diff / sq_comp)^2.
    :param ds_combined: Combined signature difference.
        Computed as sq_diff^2 / (sq_ref * sq_comp).
    """

    is_profile_comparison: bool = True
    pixel_size_ref: float = field(default=np.nan)
    pixel_size_comp: float = field(default=np.nan)
    position_shift: float = field(default=np.nan)
    scale_factor: float = field(default=np.nan)
    similarity_value: float = field(default=np.nan)
    overlap_length: float = field(default=np.nan)
    overlap_ratio: float = field(default=np.nan)
    correlation_coefficient: float = field(default=np.nan)
    sa_ref: float = field(default=np.nan)
    sq_ref: float = field(default=np.nan)
    sa_comp: float = field(default=np.nan)
    sq_comp: float = field(default=np.nan)
    sa_diff: float = field(default=np.nan)
    sq_diff: float = field(default=np.nan)
    ds_ref_norm: float = field(default=np.nan)
    ds_comp_norm: float = field(default=np.nan)
    ds_combined: float = field(default=np.nan)
