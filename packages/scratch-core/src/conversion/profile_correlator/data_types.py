"""
Data types for profile correlation.

This module defines the core data structures used for profile correlation analysis
of striated marks. It follows the patterns established in preprocess_impression.

The main types are:
- Profile: Container for 1D profile data with metadata
- AlignmentParameters: Configuration for the brute-force alignment algorithm
- StriationComparisonResults: Full comparison metrics for striated mark analysis

All length and height measurements are in meters (SI units).
"""

from dataclasses import dataclass

from container_models.base import FloatArray1D
from conversion.data_formats import Mark


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
    :param n_scale_steps: Number of steps along each scale factor. The scaling procedure is made symmetric resulting in (2n - 1) different scaling elements.
    :param min_overlap_distance: Minimum required overlap between profiles
        in meters. Alignments with less overlap are rejected.
    """

    max_scaling: float = 0.05
    n_scale_steps: int = 7
    min_overlap_distance: float = 350e-6  # 350 μm


@dataclass(frozen=True)
class RoughnessMetrics:
    """
    Container for square-based roughness metrics of a profile pair.

    Uses ISO 25178 naming conventions:
    - Sq: Quadratic mean roughness ('S' = surface, 'q' = quadratic mean)

    :param mean_square_ref: mean square 'roughness' of the reference profile.
    :param mean_square_comp: mean square 'roughness' of the comparison profile.
    :param mean_square_of_difference: mean square 'roughness' of the difference profile.
    """

    mean_square_ref: float
    mean_square_comp: float
    mean_square_of_difference: float


@dataclass(frozen=True)
class NormalizedSquareBasedRoughnessDifferences:
    """
    Container for normalized square-based roughness difference metrics.

    These metrics quantify the difference between profiles normalized by
    their roughness, providing dimensionless measures of dissimilarity.

    :param roughness_normalized_to_reference: square-based roughness difference normalized to reference,
        computed as (mean_square_of_difference / mean_square_ref)^2.
    :param roughness_normalized_to_compared: square-based roughness difference normalized to comparison,
        computed as (mean_square_of_difference / mean_square_comp)^2.
    :param roughness_normalized_to_reference_and_compared: square-based roughness difference using geometric mean
        normalization, computed as mean_square_of_difference^2 / (mean_square_ref * mean_square_comp).
    """

    roughness_normalized_to_reference: float
    roughness_normalized_to_compared: float
    roughness_normalized_to_reference_and_compared: float


@dataclass(frozen=True)
class AlignmentInputs:
    """Prepared inputs for the alignment search."""

    heights_ref: FloatArray1D
    heights_comp: FloatArray1D
    pixel_size: float
    scale_factors: FloatArray1D
    min_overlap_samples: int


@dataclass(frozen=True)
class AlignmentResult:
    """Result from the alignment search."""

    correlation: float
    shift: int
    scale: float
    ref_overlap: FloatArray1D
    comp_overlap: FloatArray1D


@dataclass(frozen=True)
class StriationComparisonResults:
    """
    Full comparison metrics for striated mark analysis.

    This immutable dataclass contains all metrics computed during profile comparison,
    including registration parameters, roughness measurements, and
    signature differences.

    All length and height measurements are in meters (SI units).

    Roughness parameters use ISO 25178 naming conventions:
    - Sa: Arithmetic mean roughness ('S' = surface, 'a' = arithmetical mean)
    - Sq: Root-mean-square roughness ('S' = surface, 'q' = quadratic mean)

    :param pixel_size: Pixel separation of profile in meters.
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
    :param mean_square_ref: Root-mean-square roughness (Sq) of the reference profile in meters.
    :param sa_comp: Arithmetic mean roughness (Sa) of the compared profile in meters.
    :param mean_square_comp: Root-mean-square roughness (Sq) of the compared profile in meters.
    :param sa_diff: Arithmetic mean roughness (Sa) of the difference profile
        (comparison minus reference) in meters.
    :param mean_square_of_difference: Root-mean-square roughness (Sq) of the difference profile
        (comparison minus reference) in meters.
    :param ds_roughness_normalized_to_reference: Signature difference normalized by reference Sq,
        computed as (mean_square_of_difference / mean_square_ref)^2.
    :param ds_roughness_normalized_to_compared: Signature difference normalized by compared Sq,
        computed as (mean_square_of_difference / mean_square_comp)^2.
    :param ds_roughness_normalized_to_reference_and_compared: Combined signature difference using geometric mean
        normalization, computed as mean_square_of_difference^2 / (mean_square_ref * mean_square_comp).
    """

    pixel_size: float
    position_shift: float
    scale_factor: float
    similarity_value: float
    overlap_length: float
    overlap_ratio: float
    correlation_coefficient: float
    sa_ref: float
    mean_square_ref: float
    sa_comp: float
    mean_square_comp: float
    sa_diff: float
    mean_square_of_difference: float
    ds_roughness_normalized_to_reference: float
    ds_roughness_normalized_to_compared: float
    ds_roughness_normalized_to_reference_and_compared: float

    @property
    def mean_square_ratio(self) -> float:
        return (self.mean_square_comp / self.mean_square_ref) * 100


@dataclass(frozen=True)
class MarkCorrelationResult:
    """
    Result of correlating two striation marks, including aligned mark regions.

    :param comparison_results: Statistical comparison metrics.
    :param mark_reference_aligned: Rows of the equalized reference mark that overlap with comp.
    :param mark_compared_aligned: Rows of the equalized, scaled comparison mark that overlap with ref.
    :param profile_reference_aligned: Reference overlap as a Profile, pixel_size = equalized pixel size.
    :param profile_compared_aligned: Comparison overlap as a Profile, pixel_size = equalized pixel size.
    """

    comparison_results: StriationComparisonResults
    mark_reference_aligned: Mark
    mark_compared_aligned: Mark
    profile_reference_aligned: Profile
    profile_compared_aligned: Profile
