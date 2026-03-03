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
    Root-mean-square roughness values for a pair of aligned profiles.

    Used as input to compute normalized signature differences. All values
    are Sq (ISO 25178) in meters.

    :param sq_ref: Root-mean-square roughness (Sq) of the reference
        overlap region, in meters.
    :param sq_comp: Root-mean-square roughness (Sq) of the compared
        overlap region, in meters.
    :param sq_diff: Root-mean-square roughness (Sq) of the difference
        profile (compared minus reference), in meters.
    """

    sq_ref: float
    sq_comp: float
    sq_diff: float


@dataclass(frozen=True)
class NormalizedRoughnessDifferences:
    """
    Normalized signature differences between two aligned profiles.

    These dimensionless metrics quantify how much the difference profile's
    roughness deviates from the individual profile roughnesses. Lower values
    indicate more similar surfaces. All three metrics equal zero when the
    profiles are identical.

    :param ds_normalized_ref: Signature difference normalized by the
        reference Sq, computed as (sq_diff / sq_ref)².
    :param ds_normalized_comp: Signature difference normalized by the
        compared Sq, computed as (sq_diff / sq_comp)².
    :param ds_normalized_combined: Signature difference using geometric
        mean normalization, computed as sq_diff² / (sq_ref * sq_comp).
        Symmetric with respect to which profile is reference vs compared.
    """

    ds_normalized_ref: float
    ds_normalized_comp: float
    ds_normalized_combined: float


@dataclass(frozen=True)
class AlignmentInputs:
    """
    Prepared inputs for the brute-force alignment search.

    Created by _prepare_alignment_inputs, this dataclass bundles the
    equalized profile heights, shared pixel size, and search parameters
    so they can be passed as a single unit to _find_best_alignment.

    :param heights_ref: Heights of the reference profile after pixel scale
        equalization, as a 1D array.
    :param heights_comp: Heights of the compared profile after pixel scale
        equalization, as a 1D array.
    :param pixel_size: Pixel separation after equalization in meters. Both
        profiles share this value.
    :param scale_factors: Array of scale factors to try during alignment.
        Generated symmetrically so that swapping reference and compared
        produces equivalent search coverage.
    :param min_overlap_samples: Minimum required overlap between profiles
        in samples, derived from AlignmentParameters.min_overlap_distance
        and the equalized pixel_size.
    :param pixel_size_reference: Original pixel size of the reference
        profile before equalization, in meters. Preserved for metadata
        in the final results.
    :param pixel_size_compared: Original pixel size of the compared
        profile before equalization, in meters. Preserved for metadata
        in the final results.
    """

    heights_ref: FloatArray1D
    heights_comp: FloatArray1D
    pixel_size: float
    scale_factors: FloatArray1D
    min_overlap_samples: int
    pixel_size_reference: float
    pixel_size_compared: float


@dataclass(frozen=True)
class AlignmentResult:
    """
    Result of the brute-force alignment search between two profiles.

    Contains the optimal registration parameters (shift, scale) and the
    corresponding overlap regions extracted from the reference and compared
    profiles. All indices and lengths are in pixels (discrete array positions).
    Multiply by pixel_size to convert to meters.

    :param correlation: Pearson cross-correlation coefficient at the best
        alignment position. Higher values indicate better agreement.
    :param shift: Optimal shift of the compared profile relative to the
        reference, in pixels. A positive value means the overlap starts at
        index ``shift`` in the reference and index 0 in the compared profile
        (i.e., the compared profile aligns with a later region of the
        reference). A negative value means the overlap starts at index 0 in
        the reference and index ``-shift`` in the compared profile.
    :param scale: Scale factor applied to the compared profile before
        alignment. Values above 1.0 stretch the compared profile (more
        pixels), below 1.0 compress it (fewer pixels). A value of 1.0
        means no resampling.
    :param ref_overlap: Heights of the reference profile within the
        overlap region, as a 1D array.
    :param comp_overlap: Heights of the scaled compared profile within the
        overlap region, as a 1D array. Same length as ref_overlap.
    :param idx_reference_start: Start index into the equalized reference
        profile where the overlap region begins. Equals ``max(shift, 0)``.
    :param idx_compared_start: Start index into the scaled compared
        profile where the overlap region begins. Equals ``max(-shift, 0)``.
    :param overlap_length: Number of pixels in the overlap region.
    """

    correlation: float
    shift: int
    scale: float
    ref_overlap: FloatArray1D
    comp_overlap: FloatArray1D
    idx_reference_start: int
    idx_compared_start: int
    overlap_length: int


@dataclass(frozen=True)
class StriationComparisonResults:
    """
    Full comparison metrics for striated mark analysis.

    This immutable dataclass contains all metrics computed during profile
    comparison, including registration parameters, roughness measurements,
    and normalized signature differences.

    All length and height measurements are in meters (SI units).

    Roughness parameters use ISO 25178 naming conventions:
    - Sa: Arithmetic mean roughness ('S' = surface, 'a' = arithmetical mean)
    - Sq: Root-mean-square roughness ('S' = surface, 'q' = quadratic mean)

    Registration
    ------------
    :param pixel_size: Pixel separation of the equalized profile in meters.
    :param position_shift: Shift of the compared profile relative to the
        reference after alignment, in meters.
    :param scale_factor: Scale factor applied to the compared profile during
        alignment (1.0 means no scaling).
    :param correlation_coefficient: Pearson cross-correlation coefficient
        between the aligned overlap regions. This is both the optimization
        target during alignment and the final similarity score.
    :param overlap_length: Length of the overlapping region after alignment,
        in meters.
    :param overlap_ratio: Ratio of overlap length to the length of the
        shorter profile (0.0–1.0).

    Roughness — individual profiles
    --------------------------------
    :param sa_ref: Arithmetic mean roughness (Sa) of the reference overlap
        region, in meters.
    :param sq_ref: Root-mean-square roughness (Sq) of the reference overlap
        region, in meters.
    :param sa_comp: Arithmetic mean roughness (Sa) of the compared overlap
        region, in meters.
    :param sq_comp: Root-mean-square roughness (Sq) of the compared overlap
        region, in meters.

    Roughness — difference profile
    ------------------------------
    :param sa_diff: Arithmetic mean roughness (Sa) of the difference profile
        (compared minus reference), in meters.
    :param sq_diff: Root-mean-square roughness (Sq) of the difference profile
        (compared minus reference), in meters.

    Normalized signature differences (dimensionless)
    -------------------------------------------------
    :param ds_normalized_ref: Signature difference normalized by reference Sq,
        computed as (sq_diff / sq_ref)².
    :param ds_normalized_comp: Signature difference normalized by compared Sq,
        computed as (sq_diff / sq_comp)².
    :param ds_normalized_combined: Signature difference using geometric mean
        normalization, computed as sq_diff² / (sq_ref * sq_comp).

    Sample-space geometry
    ---------------------
    :param shift_samples: Alignment shift in samples (integer). Avoids
        back-conversion from position_shift via rounding.
    :param overlap_samples: Number of samples in the overlap region.
    :param idx_reference_start: Start index of the overlap within the
        equalized reference profile.
    :param idx_compared_start: Start index of the overlap within the
        scaled compared profile.
    :param len_reference_equalized: Total length of the equalized reference
        profile in samples.
    :param len_compared_equalized: Total length of the equalized compared
        profile in samples (before scaling).

    Original profile metadata
    -------------------------
    :param pixel_size_reference: Original pixel size of the reference profile
        before equalization, in meters.
    :param pixel_size_compared: Original pixel size of the compared profile
        before equalization, in meters.
    :param len_reference_original: Length of the reference profile in samples
        as provided to correlate_profiles.
    :param len_compared_original: Length of the compared profile in samples
        as provided to correlate_profiles.

    Reproducibility
    ---------------
    :param alignment_parameters: The AlignmentParameters used for this
        comparison, stored for provenance and reproducibility.
    """

    # Registration
    pixel_size: float
    position_shift: float
    scale_factor: float
    correlation_coefficient: float
    overlap_length: float
    overlap_ratio: float

    # Roughness — individual profiles
    sa_ref: float
    sq_ref: float
    sa_comp: float
    sq_comp: float

    # Roughness — difference profile
    sa_diff: float
    sq_diff: float

    # Normalized signature differences
    ds_normalized_ref: float
    ds_normalized_comp: float
    ds_normalized_combined: float

    # Sample-space geometry
    shift_samples: int
    overlap_samples: int
    idx_reference_start: int
    idx_compared_start: int
    len_reference_equalized: int
    len_compared_equalized: int

    # Original profile metadata
    pixel_size_reference: float
    pixel_size_compared: float
    len_reference_original: int
    len_compared_original: int

    # Reproducibility
    alignment_parameters: AlignmentParameters

    @property
    def sq_ratio(self) -> float:
        """Ratio of compared to reference Sq, as a percentage."""
        return (self.sq_comp / self.sq_ref) * 100


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
