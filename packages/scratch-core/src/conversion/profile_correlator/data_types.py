"""Data types for profile correlation.

This module defines the core data structures used for profile correlation analysis
of striated marks. It follows the patterns established in preprocess_impression.

The main types are:
- Profile: Container for 1D or multi-column profile data with metadata
- AlignmentParameters: Configuration for the alignment algorithm
- TransformParameters: Single translation + scaling transform
- AlignmentResult: Complete result from multi-scale alignment
- ComparisonResults: Full comparison metrics for striated mark analysis
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Profile:
    """
    Container for a 1D or multi-column profile with metadata.

    Profiles represent 1D height measurements along a scratch/striation mark.
    Multi-column profiles contain multiple parallel scans that can be averaged
    using mean or median to reduce noise.

    The MATLAB equivalent is a struct with fields:
    - depth_data: Height values (N,) or (N, M) array
    - xdim: Physical distance between samples in meters (pixel_size)
    - cutoff_hi, cutoff_lo: Filter cutoff wavelengths in micrometers
    - LR: Minimum resolvable wavelength (resolution_limit)

    :param depth_data: Height values as (N,) for single profile or (N, M)
        for M parallel profiles where N is the number of samples.
    :param pixel_size: Physical distance between samples in meters.
        This corresponds to 'xdim' in the MATLAB code.
    :param cutoff_hi: High-frequency cutoff wavelength in micrometers.
        Wavelengths shorter than this are filtered out (optional).
    :param cutoff_lo: Low-frequency cutoff wavelength in micrometers.
        Wavelengths longer than this are filtered out (optional).
    :param resolution_limit: Minimum resolvable wavelength in meters.
        This corresponds to 'LR' in the MATLAB code (optional).
    """

    depth_data: NDArray[np.floating]
    pixel_size: float
    cutoff_hi: float | None = None
    cutoff_lo: float | None = None
    resolution_limit: float | None = None

    @property
    def length(self) -> int:
        """
        Get the number of samples in the profile.

        :returns: Number of samples (first dimension of depth_data).
        """
        return self.depth_data.shape[0]

    @property
    def num_columns(self) -> int:
        """
        Get the number of parallel profile columns.

        :returns: 1 for a single profile, M for multi-column profiles.
        """
        return 1 if self.depth_data.ndim == 1 else self.depth_data.shape[1]

    @property
    def pixel_size_um(self) -> float:
        """
        Get the pixel size in micrometers.

        :returns: Pixel size converted from meters to micrometers.
        """
        return self.pixel_size * 1e6

    def mean_profile(self, use_mean: bool = True) -> NDArray[np.floating]:
        """
        Compute mean or median across columns.

        For multi-column profiles, this reduces to a single 1D profile by
        averaging across the columns. NaN values are ignored in the computation.

        :param use_mean: If True, use nanmean; if False, use nanmedian.
        :returns: 1D array of the averaged profile.
        """
        if self.depth_data.ndim == 1:
            return self.depth_data
        func = np.nanmean if use_mean else np.nanmedian
        return func(self.depth_data, axis=1)


@dataclass
class AlignmentParameters:
    """
    Configuration parameters for profile alignment.

    This dataclass contains all parameters needed for the multi-scale
    profile alignment algorithm. Default values match the MATLAB implementation.

    :param scale_passes: Cutoff wavelengths (um) for multi-scale passes,
        ordered from coarse to fine. At each scale, profiles are low-pass
        filtered and aligned before proceeding to the next finer scale.
    :param max_translation: Maximum allowed translation in micrometers.
        The optimization will not exceed this shift distance.
    :param max_scaling: Maximum allowed scaling deviation as a fraction.
        E.g., 0.05 means scaling can vary from 0.95 to 1.05.
    :param cutoff_hi: High-frequency cutoff for filtering in micrometers.
        Scales finer than this will not be used.
    :param cutoff_lo: Low-frequency cutoff for filtering in micrometers.
        Scales coarser than this will not be used.
    :param partial_mark_threshold: Length difference threshold (percent)
        to trigger partial profile matching. If profiles differ by more
        than this percentage, brute-force candidate search is used.
    :param inclusion_threshold: Minimum correlation coefficient for
        accepting a candidate position in partial profile matching.
    :param use_mean: If True, average multi-column profiles with mean;
        if False, use median.
    :param remove_boundary_zeros: If True, remove zero-padded boundaries
        after alignment transformation.
    :param cut_borders_after_smoothing: If True, trim filter-affected
        borders after applying smoothing filters.
    """

    scale_passes: tuple[float, ...] = (1000, 500, 250, 100, 50, 25, 10, 5)
    max_translation: float = 1e7
    max_scaling: float = 0.05
    cutoff_hi: float = 1000.0
    cutoff_lo: float = 5.0
    partial_mark_threshold: float = 8.0
    inclusion_threshold: float = 0.5
    use_mean: bool = True
    remove_boundary_zeros: bool = True
    cut_borders_after_smoothing: bool = False


@dataclass(frozen=True)
class TransformParameters:
    """
    Single translation and scaling transformation parameters.

    This immutable dataclass represents one step in the multi-scale
    alignment process. Each scale level produces one set of transform
    parameters.

    The transformation is applied as: x' = scaling * x + translation

    :param translation: Shift distance in samples (can be fractional
        after optimization).
    :param scaling: Scale factor where 1.0 means no scaling.
    """

    translation: float
    scaling: float


@dataclass(frozen=True)
class AlignmentResult:
    """
    Complete result from profile alignment.

    This immutable dataclass contains all outputs from the multi-scale
    alignment process, including the sequence of transforms applied,
    correlation history, and the final aligned profiles.

    :param transforms: Tuple of TransformParameters, one per scale level
        that was processed. Ordered from coarse to fine.
    :param correlation_history: Array of shape (num_scales, 2) where
        column 0 is the correlation at each scale level and column 1
        is the correlation of the original (unfiltered) profiles after
        applying the cumulative transform up to that scale.
    :param final_correlation: Final cross-correlation coefficient
        between the aligned profiles.
    :param reference_aligned: Reference profile after removing boundary
        zeros (if enabled).
    :param compared_aligned: Compared profile after alignment
        transformation and boundary zero removal.
    :param total_translation: Cumulative translation in samples,
        computed from all transform steps.
    :param total_scaling: Cumulative scaling factor, computed as the
        product of all scaling values.
    """

    transforms: tuple[TransformParameters, ...]
    correlation_history: NDArray[np.floating]
    final_correlation: float
    reference_aligned: NDArray[np.floating]
    compared_aligned: NDArray[np.floating]
    total_translation: float
    total_scaling: float


@dataclass(frozen=True)
class ComparisonResults:
    """
    Full comparison metrics for striated mark analysis.

    This immutable dataclass mirrors the MATLAB ProfileCorrelatorResInit
    structure. It contains all metrics computed during profile comparison,
    including registration parameters, roughness measurements, and
    signature differences.

    All length measurements are in micrometers unless otherwise noted.
    Height measurements (Sa, Sq) are also in micrometers.

    :param is_profile_comparison: True for full profile comparison mode.
    :param is_partial_profile: True if partial profile matching was used
        (profiles had significantly different lengths).
    :param pixel_size_ref: Pixel separation of reference profile (um).
    :param pixel_size_comp: Pixel separation of compared profile (um).
    :param position_shift: Registration shift of compared profile
        relative to reference (um).
    :param scale_factor: Registration scale factor applied to compared
        profile (1.0 = no scaling).
    :param partial_start_position: For partial profile matching, the
        position in the reference where the partial profile best aligns
        (um). NaN for full profile matching.
    :param similarity_value: Optimized value of the similarity metric
        used during registration.
    :param overlap_length: Length of the overlapping region after
        registration (um).
    :param overlap_ratio: Ratio of overlap length to the length of the
        shorter profile.
    :param correlation_coefficient: Pearson cross-correlation coefficient
        between the aligned profiles.
    :param sa_ref: Mean absolute height (Sa) of the reference profile (um).
    :param sq_ref: RMS roughness (Sq) of the reference profile (um).
    :param sa_comp: Mean absolute height (Sa) of the compared profile (um).
    :param sq_comp: RMS roughness (Sq) of the compared profile (um).
    :param sa_diff: Mean absolute height difference between profiles (um).
    :param sq_diff: RMS height difference between profiles (um).
    :param ds_ref_norm: Signature difference normalized by reference Sq.
        Computed as (sq_diff / sq_ref)^2.
    :param ds_comp_norm: Signature difference normalized by compared Sq.
        Computed as (sq_diff / sq_comp)^2.
    :param ds_combined: Combined signature difference.
        Computed as sq_diff^2 / (sq_ref * sq_comp).
    """

    is_profile_comparison: bool = True
    is_partial_profile: bool = False
    pixel_size_ref: float = field(default=np.nan)
    pixel_size_comp: float = field(default=np.nan)
    position_shift: float = field(default=np.nan)
    scale_factor: float = field(default=np.nan)
    partial_start_position: float = field(default=np.nan)
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
