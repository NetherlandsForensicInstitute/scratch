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

from dataclasses import dataclass

from container_models.base import FloatArray1D


@dataclass(frozen=True)
class Profile:
    """
    Container for a 1D profile with metadata.

    Profiles represent 1D height measurements along a scratch/striation mark.
    All measurements are in meters (SI units).

    :param heights: Height values as a 1D array of shape (N,).
    :param pixel_size: Physical distance between samples in meters.
    """

    heights: FloatArray1D
    pixel_size: float

    @property
    def length(self) -> int:
        """
        Get the number of samples in the profile.

        :returns: Number of samples in heights.
        """
        return len(self.heights)


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
class RoughnessMetrics:
    """
    Container for roughness metrics of a profile pair.

    Uses ISO 25178 naming conventions:
    - Sa: Arithmetic mean roughness ('S' = surface, 'a' = arithmetical mean)
    - Sq: Quadratic mean roughness ('S' = surface, 'q' = quadratic mean)

    :param sq_ref: Quadratic mean roughness (Sq) of the reference profile.
    :param sq_comp: Quadratic mean roughness (Sq) of the comparison profile.
    :param sq_diff: Quadratic mean roughness (Sq) of the difference profile
        (comparison minus reference).
    """

    sq_ref: float
    sq_comp: float
    sq_diff: float


@dataclass(frozen=True)
class SignatureDifferences:
    """
    Container for normalized signature difference metrics.

    These metrics quantify the difference between profiles normalized by
    their roughness, providing dimensionless measures of dissimilarity.

    :param ref_norm: Signature difference normalized to reference,
        computed as (sq_diff / sq_ref)^2.
    :param comp_norm: Signature difference normalized to comparison,
        computed as (sq_diff / sq_comp)^2.
    :param combined: Combined signature difference using geometric mean
        normalization, computed as sq_diff^2 / (sq_ref * sq_comp).
    """

    ref_norm: float
    comp_norm: float
    combined: float


@dataclass(frozen=True)
class AlignmentResult:
    """Result from the alignment search."""

    correlation: float
    shift: int
    scale: float
    ref_overlap: FloatArray1D
    comp_overlap: FloatArray1D


@dataclass(frozen=True)
class ComparisonResults:
    """
    Full comparison metrics for striated mark analysis.

    This immutable dataclass contains all metrics computed during profile comparison,
    including registration parameters, roughness measurements, and
    signature differences.

    All length and height measurements are in meters (SI units).

    Roughness parameters use ISO 25178 naming conventions:
    - Sa: Arithmetic mean roughness ('S' = surface, 'a' = arithmetical mean)
    - Sq: Root-mean-square roughness ('S' = surface, 'q' = quadratic mean)

    :param is_profile_comparison: True for full profile comparison mode.
    :param pixel_size_ref: Pixel separation of reference profile in meters.
    :param pixel_size_comp: Pixel separation of compared profile in meters.
    :param position_shift: Registration shift of compared profile relative to
        reference in meters.
    :param scale_factor: Registration scale factor applied to compared profile
        (1.0 means no scaling).
    :param similarity_value: Optimized value of the similarity metric used
        during registration.
    :param overlap_length: Length of the overlapping region after registration
        in meters.
    :param overlap_ratio: Ratio of overlap length to the length of the shorter
        profile.
    :param correlation_coefficient: Pearson cross-correlation coefficient
        between the aligned profiles.
    :param sa_ref: Arithmetic mean roughness (Sa) of the reference profile in meters.
    :param sq_ref: Root-mean-square roughness (Sq) of the reference profile in meters.
    :param sa_comp: Arithmetic mean roughness (Sa) of the compared profile in meters.
    :param sq_comp: Root-mean-square roughness (Sq) of the compared profile in meters.
    :param sa_diff: Arithmetic mean roughness (Sa) of the difference profile
        (comparison minus reference) in meters.
    :param sq_diff: Root-mean-square roughness (Sq) of the difference profile
        (comparison minus reference) in meters.
    :param ds_ref_norm: Signature difference normalized by reference Sq,
        computed as (sq_diff / sq_ref)^2.
    :param ds_comp_norm: Signature difference normalized by compared Sq,
        computed as (sq_diff / sq_comp)^2.
    :param ds_combined: Combined signature difference using geometric mean
        normalization, computed as sq_diff^2 / (sq_ref * sq_comp).
    """

    is_profile_comparison: bool
    pixel_size_ref: float
    pixel_size_comp: float
    position_shift: float
    scale_factor: float
    similarity_value: float
    overlap_length: float
    overlap_ratio: float
    correlation_coefficient: float
    sa_ref: float
    sq_ref: float
    sa_comp: float
    sq_comp: float
    sa_diff: float
    sq_diff: float
    ds_ref_norm: float
    ds_comp_norm: float
    ds_combined: float
