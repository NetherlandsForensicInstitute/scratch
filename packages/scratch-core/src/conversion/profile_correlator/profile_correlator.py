"""
Profile Correlator - Python translation from MATLAB with Gaussian filtering integration

This module provides profile registration and correlation for surface texture analysis,
particularly for forensic toolmark comparison.

Author: Translated from Martin Baiker-Soerensen's MATLAB code (NFI, Mar 2021)
Python Translation: January 2026
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.signal import resample
from scipy.interpolate import interp1d

# Import filtering functions
from conversion.filter.gaussian import (
    apply_striation_preserving_filter_1d,
)
from container_models.scan_image import ScanImage


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class Profile:
    """
    Profile data structure for 1D surface profiles.

    Attributes:
        depth_data: Height/depth measurements in meters, shape (N,) or (N, n_columns)
        pixel_size: Sampling distance in meters
        cutoff_hi: Optional high-pass cutoff in meters
        cutoff_lo: Optional low-pass cutoff in meters
    """

    depth_data: NDArray[np.floating]
    pixel_size: float
    cutoff_hi: Optional[float] = None
    cutoff_lo: Optional[float] = None

    def __post_init__(self):
        """Validate and reshape data."""
        if self.depth_data.ndim == 1:
            self.depth_data = self.depth_data.reshape(-1, 1)
        elif self.depth_data.ndim != 2:
            raise ValueError(
                f"depth_data must be 1D or 2D, got shape {self.depth_data.shape}"
            )

    @property
    def length(self) -> int:
        """Number of samples in profile."""
        return self.depth_data.shape[0]

    @property
    def n_columns(self) -> int:
        """Number of parallel scans/columns."""
        return self.depth_data.shape[1]

    def mean_profile(self) -> NDArray[np.floating]:
        """Get mean across columns."""
        if self.n_columns == 1:
            return self.depth_data.flatten()
        return np.nanmean(self.depth_data, axis=1)

    def median_profile(self) -> NDArray[np.floating]:
        """Get median across columns."""
        if self.n_columns == 1:
            return self.depth_data.flatten()
        return np.nanmedian(self.depth_data, axis=1)


@dataclass
class AlignmentParameters:
    """Parameters for profile alignment."""

    scale_passes: Tuple[float, ...] = (
        1e-3,
        5e-4,
        2.5e-4,
        1e-4,
        5e-5,
        2.5e-5,
        1e-5,
        5e-6,
    )
    max_translation: float = 10.0  # meters
    max_scaling: float = 0.05  # fraction
    cutoff_hi: float = 1e-3  # meters (1 mm)
    cutoff_lo: float = 5e-6  # meters (5 μm)
    partial_mark_threshold: float = 8.0  # percentage
    inclusion_threshold: float = 0.5
    use_mean: bool = True
    remove_boundary_zeros: bool = True
    show_info: bool = False
    initial_guess: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ComparisonResults:
    """Results from profile comparison."""

    # Flags
    is_profile_comparison: bool = False
    is_partial_profile: bool = False

    # Pixel separations (meters)
    pixel_size_ref: float = 0.0
    pixel_size_comp: float = 0.0

    # Registration results
    position_shift: float = 0.0  # meters
    scale_factor: float = 1.0
    partial_profile_start: Optional[float] = None  # meters

    # Overlap information
    overlap_length: float = 0.0  # meters
    overlap_ratio: float = 0.0

    # Similarity metrics
    correlation_coefficient: float = 0.0

    # Topographic measurements (micrometers)
    sa_ref: float = 0.0
    sq_ref: float = 0.0
    sa_comp: float = 0.0
    sq_comp: float = 0.0
    sa_diff: float = 0.0
    sq_diff: float = 0.0

    # Signature differences
    ds1: float = 0.0
    ds2: float = 0.0
    ds: float = 0.0

    # Transformation parameters (for multi-scale)
    transformation_array: Optional[NDArray[np.floating]] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_param_value(params: AlignmentParameters, attr_name: str, default):
    """Get parameter value with default fallback."""
    return getattr(params, attr_name, default)


def get_alicona_sampling() -> float:
    """Default Alicona sampling distance."""
    return 4.38312e-07


# ============================================================================
# SIMILARITY SCORING
# ============================================================================


def compute_similarity_score(
    profile_1: NDArray[np.floating],
    profile_2: NDArray[np.floating],
    score_type: str = "cross_correlation",
) -> float:
    """
    Compute similarity score between two profiles.

    Args:
        profile_1: Reference profile
        profile_2: Compared profile
        score_type: Type of similarity metric

    Returns:
        Similarity score
    """
    # Ensure 1D arrays
    p1 = profile_1.flatten()
    p2 = profile_2.flatten()

    if score_type == "cross_correlation":
        # Remove NaN values
        valid_mask = ~(np.isnan(p1) | np.isnan(p2))
        p1_valid = p1[valid_mask]
        p2_valid = p2[valid_mask]

        if len(p1_valid) < 2:
            return 0.0

        # Subtract means
        p1_centered = p1_valid - np.mean(p1_valid)
        p2_centered = p2_valid - np.mean(p2_valid)

        # Compute correlation
        numerator = np.dot(p1_centered, p2_centered)
        denominator = np.sqrt(
            np.dot(p1_centered, p1_centered) * np.dot(p2_centered, p2_centered)
        )

        if denominator == 0:
            return 0.0

        return numerator / denominator
    else:
        raise ValueError(f"Similarity metric '{score_type}' not implemented")


# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================


def apply_transformation(
    data: NDArray[np.floating], translation: float, scaling: float
) -> NDArray[np.floating]:
    """
    Apply translation and scaling transformation to profile.

    Args:
        data: Input profile (1D)
        translation: Translation in samples
        scaling: Scaling factor

    Returns:
        Transformed profile
    """
    n = len(data)
    x_orig = np.arange(n, dtype=float)
    x_transformed = x_orig * scaling + translation

    # Interpolate
    interpolator = interp1d(
        x_transformed, data, kind="linear", bounds_error=False, fill_value=0.0
    )

    return interpolator(x_orig)


def apply_multi_scale_transformation(
    data: NDArray[np.floating], trans_array: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Apply multiple transformation stages.

    Args:
        data: Input profile
        trans_array: Array of [translation, scaling] pairs, shape (n_stages, 2)

    Returns:
        Transformed profile
    """
    result = data.copy()

    for i in range(trans_array.shape[0]):
        translation = trans_array[i, 0]
        scaling = trans_array[i, 1]
        result = apply_transformation(result, translation, scaling)

    return result


def remove_boundary_zeros(
    profile_1: NDArray[np.floating], profile_2: NDArray[np.floating]
) -> Tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """
    Remove padded zeros at boundaries.

    Args:
        profile_1: First profile
        profile_2: Second profile

    Returns:
        Tuple of (cropped_profile_1, cropped_profile_2, start_index)
    """
    # Find non-zero regions
    nonzero_1 = profile_1 != 0
    nonzero_2 = profile_2 != 0

    # Find start and end indices
    indices_1 = np.where(nonzero_1)[0]
    indices_2 = np.where(nonzero_2)[0]

    if len(indices_1) == 0 or len(indices_2) == 0:
        return profile_1, profile_2, 0

    start = max(indices_1[0], indices_2[0])
    end = min(indices_1[-1], indices_2[-1])

    if start >= end:
        return profile_1, profile_2, 0

    return profile_1[start : end + 1], profile_2[start : end + 1], start


# ============================================================================
# RESAMPLING AND EQUALIZATION
# ============================================================================


def equalize_pixel_sizes(
    profile_ref: Profile, profile_comp: Profile
) -> Tuple[Profile, Profile]:
    """
    Resample profiles to have equal pixel sizes.

    Args:
        profile_ref: Reference profile
        profile_comp: Comparison profile

    Returns:
        Tuple of (resampled_ref, resampled_comp)
    """
    if profile_ref.pixel_size == profile_comp.pixel_size:
        return profile_ref, profile_comp

    # Determine which needs resampling
    if profile_ref.pixel_size > profile_comp.pixel_size:
        # Resample comp to match ref
        resample_factor = profile_comp.pixel_size / profile_ref.pixel_size
        new_length = int(profile_comp.length * resample_factor)

        resampled_data = resample(profile_comp.depth_data, new_length, axis=0)

        profile_comp_new = Profile(
            depth_data=resampled_data,
            pixel_size=profile_ref.pixel_size,
            cutoff_hi=profile_comp.cutoff_hi,
            cutoff_lo=profile_comp.cutoff_lo,
        )

        return profile_ref, profile_comp_new
    else:
        # Resample ref to match comp
        resample_factor = profile_ref.pixel_size / profile_comp.pixel_size
        new_length = int(profile_ref.length * resample_factor)

        resampled_data = resample(profile_ref.depth_data, new_length, axis=0)

        profile_ref_new = Profile(
            depth_data=resampled_data,
            pixel_size=profile_comp.pixel_size,
            cutoff_hi=profile_ref.cutoff_hi,
            cutoff_lo=profile_ref.cutoff_lo,
        )

        return profile_ref_new, profile_comp


def make_equal_length(
    profile_1: NDArray[np.floating], profile_2: NDArray[np.floating]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Crop profiles to equal length by removing from ends.

    Args:
        profile_1: First profile
        profile_2: Second profile

    Returns:
        Tuple of equal-length profiles
    """
    len1 = len(profile_1)
    len2 = len(profile_2)

    if len1 == len2:
        return profile_1, profile_2

    min_len = min(len1, len2)

    # Crop from center
    start1 = (len1 - min_len) // 2
    end1 = start1 + min_len

    start2 = (len2 - min_len) // 2
    end2 = start2 + min_len

    return profile_1[start1:end1], profile_2[start2:end2]


# ============================================================================
# FILTERING
# ============================================================================


def apply_lowpass_filter(
    data: NDArray[np.floating], cutoff: float, pixel_size: float
) -> NDArray[np.floating]:
    """
    Apply 1D lowpass filter to profile.

    Args:
        data: Profile data (1D)
        cutoff: Cutoff wavelength in meters
        pixel_size: Pixel spacing in meters

    Returns:
        Filtered profile
    """
    # Create a ScanImage for the filtering function
    # Reshape to 2D if needed
    if data.ndim == 1:
        data_2d = data.reshape(-1, 1)
    else:
        data_2d = data

    scan_image = ScanImage(data=data_2d, scale_x=pixel_size, scale_y=pixel_size)

    # Apply striation-preserving filter (1D along y-direction)
    filtered_data, _ = apply_striation_preserving_filter_1d(
        scan_image, cutoff=cutoff, is_high_pass=False, cut_borders_after_smoothing=False
    )

    # Return as 1D
    if filtered_data.shape[1] == 1:
        return filtered_data.flatten()
    return filtered_data


# ============================================================================
# OPTIMIZATION
# ============================================================================


def error_function(
    params: NDArray[np.floating],
    profile_ref: NDArray[np.floating],
    profile_comp: NDArray[np.floating],
) -> float:
    """
    Error function for optimization.

    Args:
        params: [translation, scaling*10000-10000]
        profile_ref: Reference profile
        profile_comp: Comparison profile

    Returns:
        Negative similarity score
    """
    translation = params[0]
    scaling = params[1] / 10000.0 + 1.0

    # Apply transformation
    profile_transformed = apply_transformation(profile_comp, translation, scaling)

    # Compute similarity (negative for minimization)
    similarity = compute_similarity_score(profile_ref, profile_transformed)

    return -similarity


class BoundedOptimizer:
    """Bounded optimization using scipy."""

    @staticmethod
    def optimize(
        func,
        x0: NDArray[np.floating],
        bounds: Tuple[Tuple[float, float], ...],
        args: tuple = (),
    ) -> Tuple[NDArray[np.floating], float]:
        """
        Perform bounded optimization.

        Args:
            func: Objective function
            x0: Initial guess
            bounds: Parameter bounds
            args: Additional arguments for func

        Returns:
            Tuple of (optimal_params, function_value)
        """
        result = minimize(
            func,
            x0,
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-6, "gtol": 1e-6},
        )

        return result.x, result.fun


# ============================================================================
# ALIGNMENT FUNCTIONS
# ============================================================================


def align_profiles_multiscale(
    profile_ref: Profile, profile_comp: Profile, params: AlignmentParameters
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Align two profiles using multi-scale registration.

    Args:
        profile_ref: Reference profile
        profile_comp: Comparison profile
        params: Alignment parameters

    Returns:
        Tuple of (trans_array, profile_ref_aligned, profile_comp_aligned, cross_correlations)
    """
    # Get mean/median profiles
    if params.use_mean:
        data_ref = profile_ref.mean_profile()
        data_comp = profile_comp.mean_profile()
    else:
        data_ref = profile_ref.median_profile()
        data_comp = profile_comp.median_profile()

    # Check lengths match
    if len(data_ref) != len(data_comp):
        raise ValueError("Profiles must have equal length for multi-scale alignment")

    # Initialize
    pixel_size = profile_ref.pixel_size
    cutoff_hi = params.cutoff_hi
    cutoff_lo = params.cutoff_lo

    trans_array = []
    cross_correlations = []

    data_comp_transformed = data_comp.copy()
    cumulative_translation = 0.0
    cumulative_scaling = 1.0

    # Multi-scale loop
    for scale_idx, cutoff in enumerate(params.scale_passes):
        # Skip if outside cutoff range
        if cutoff > cutoff_hi or cutoff < cutoff_lo:
            continue

        # Check resolution threshold
        resolution_threshold = max(2 * pixel_size, cutoff_lo)
        if cutoff < resolution_threshold:
            continue

        if params.show_info:
            print(f"Scale {scale_idx + 1}: cutoff = {cutoff * 1e6:.1f} μm")

        # Apply lowpass filter
        filtered_ref = apply_lowpass_filter(data_ref, cutoff, pixel_size)
        filtered_comp = apply_lowpass_filter(data_comp_transformed, cutoff, pixel_size)

        # Subsample for efficiency
        subsample_factor = max(1, int(np.ceil(cutoff / pixel_size / 2 / 5)))
        subsampled_ref = filtered_ref[::subsample_factor]
        subsampled_comp = filtered_comp[::subsample_factor]

        # Set up optimization bounds
        max_trans = params.max_translation / pixel_size
        max_scale = params.max_scaling

        max_trans_samples = max_trans / subsample_factor

        bounds = (
            (-max_trans_samples, max_trans_samples),
            (
                (-max_scale / cumulative_scaling) * 10000,
                (max_scale / cumulative_scaling) * 10000,
            ),
        )

        # Initial guess
        x0 = np.array([0.0, 0.0])

        # Optimize
        optimal_params, _ = BoundedOptimizer.optimize(
            error_function, x0, bounds, args=(subsampled_ref, subsampled_comp)
        )

        # Extract transformation
        translation = optimal_params[0] * subsample_factor
        scaling = optimal_params[1] / 10000.0 + 1.0

        # Update cumulative transformation
        cumulative_translation += translation
        cumulative_scaling *= scaling

        # Store
        trans_array.append([translation, scaling])

        # Apply transformation to full-resolution data
        data_comp_transformed = apply_transformation(data_comp, translation, scaling)

        # Compute correlation
        corr = compute_similarity_score(
            filtered_ref, apply_transformation(filtered_comp, translation, scaling)
        )
        full_corr = compute_similarity_score(data_ref, data_comp_transformed)
        cross_correlations.append([corr, full_corr])

    # Convert to arrays
    trans_array = np.array(trans_array) if trans_array else np.array([[0.0, 1.0]])
    cross_correlations = (
        np.array(cross_correlations) if cross_correlations else np.array([[0.0, 0.0]])
    )

    # Remove boundary zeros if requested
    if params.remove_boundary_zeros:
        data_ref_out, data_comp_out, _ = remove_boundary_zeros(
            data_ref, data_comp_transformed
        )
    else:
        data_ref_out = data_ref
        data_comp_out = data_comp_transformed

    return trans_array, data_ref_out, data_comp_out, cross_correlations


def align_partial_profile(
    profile_ref: Profile, profile_partial: Profile, params: AlignmentParameters
) -> Tuple[
    NDArray[np.floating], float, float, NDArray[np.floating], NDArray[np.floating]
]:
    """
    Align a partial profile to a full reference profile.

    Args:
        profile_ref: Full reference profile
        profile_partial: Partial profile
        params: Alignment parameters

    Returns:
        Tuple of (trans_array, best_position, correlation, aligned_ref, aligned_comp)
    """
    # Get mean/median
    if params.use_mean:
        data_ref = profile_ref.mean_profile()
        data_partial = profile_partial.mean_profile()
    else:
        data_ref = profile_ref.median_profile()
        data_partial = profile_partial.median_profile()

    len_ref = len(data_ref)
    len_partial = len(data_partial)

    if len_partial >= len_ref:
        raise ValueError("Partial profile must be shorter than reference")

    # Brute force search for best position
    best_corr = -np.inf
    best_pos = 0
    best_trans = np.array([[0.0, 1.0]])
    best_ref_segment = None
    best_aligned = None

    for pos in range(len_ref - len_partial + 1):
        # Extract segment
        ref_segment_data = data_ref[pos : pos + len_partial]

        # Create temporary profile
        ref_segment = Profile(
            depth_data=ref_segment_data.reshape(-1, 1),
            pixel_size=profile_ref.pixel_size,
            cutoff_hi=profile_ref.cutoff_hi,
            cutoff_lo=profile_ref.cutoff_lo,
        )

        partial_temp = Profile(
            depth_data=data_partial.reshape(-1, 1),
            pixel_size=profile_partial.pixel_size,
            cutoff_hi=profile_partial.cutoff_hi,
            cutoff_lo=profile_partial.cutoff_lo,
        )

        # Align this segment
        try:
            trans, ref_aligned, partial_aligned, xcorr = align_profiles_multiscale(
                ref_segment, partial_temp, params
            )

            corr = xcorr[-1, 1] if len(xcorr) > 0 else 0.0

            if corr > best_corr:
                best_corr = corr
                best_pos = pos
                best_trans = trans
                best_ref_segment = ref_aligned
                best_aligned = partial_aligned

        except Exception as e:
            if params.show_info:
                print(f"Position {pos} failed: {e}")
            continue

    return (
        best_trans,
        float(best_pos * profile_ref.pixel_size),
        best_corr,
        best_ref_segment,
        best_aligned,
    )


# ============================================================================
# MAIN CORRELATION FUNCTION
# ============================================================================


def correlate_profiles(
    profile_ref: Profile,
    profile_comp: Profile,
    params: Optional[AlignmentParameters] = None,
) -> ComparisonResults:
    """
    Main function to correlate two profiles.

    Args:
        profile_ref: Reference profile
        profile_comp: Comparison profile
        params: Alignment parameters (uses defaults if None)

    Returns:
        ComparisonResults object with all comparison metrics
    """
    if params is None:
        params = AlignmentParameters()

    # Initialize results
    results = ComparisonResults()
    results.is_profile_comparison = True

    # Step 1: Equalize pixel sizes
    prof_ref_eq, prof_comp_eq = equalize_pixel_sizes(profile_ref, profile_comp)

    results.pixel_size_ref = prof_ref_eq.pixel_size
    results.pixel_size_comp = prof_comp_eq.pixel_size

    # Step 2: Check if partial profile comparison
    # IMPORTANT: Use equalized lengths for determining partial profile and overlap calculations
    len_ref_eq = prof_ref_eq.length
    len_comp_eq = prof_comp_eq.length

    length_diff_pct = (
        abs(len_ref_eq - len_comp_eq) / max(len_ref_eq, len_comp_eq) * 100.0
    )

    if length_diff_pct < params.partial_mark_threshold:
        # Full profile comparison
        results.is_partial_profile = False

        # Make equal length
        data_ref = (
            prof_ref_eq.mean_profile()
            if params.use_mean
            else prof_ref_eq.median_profile()
        )
        data_comp = (
            prof_comp_eq.mean_profile()
            if params.use_mean
            else prof_comp_eq.median_profile()
        )

        data_ref, data_comp = make_equal_length(data_ref, data_comp)

        # Create temporary profiles with equal length
        prof_ref_temp = Profile(
            depth_data=data_ref.reshape(-1, 1),
            pixel_size=prof_ref_eq.pixel_size,
            cutoff_hi=prof_ref_eq.cutoff_hi,
            cutoff_lo=prof_ref_eq.cutoff_lo,
        )

        prof_comp_temp = Profile(
            depth_data=data_comp.reshape(-1, 1),
            pixel_size=prof_comp_eq.pixel_size,
            cutoff_hi=prof_comp_eq.cutoff_hi,
            cutoff_lo=prof_comp_eq.cutoff_lo,
        )

        # Align
        trans_array, data_ref_aligned, data_comp_aligned, xcorr = (
            align_profiles_multiscale(prof_ref_temp, prof_comp_temp, params)
        )

        results.transformation_array = trans_array
        results.correlation_coefficient = xcorr[-1, 1] if len(xcorr) > 0 else 0.0

        # Compute transformation parameters
        cumulative_trans = np.eye(3)
        for i in range(trans_array.shape[0]):
            t = trans_array[i, 0]
            s = trans_array[i, 1]
            trans_mat = np.array([[s, 0, t], [0, 1, 0], [0, 0, 1]])
            cumulative_trans = trans_mat @ cumulative_trans

        results.position_shift = cumulative_trans[0, 2] * results.pixel_size_ref
        results.scale_factor = cumulative_trans[0, 0]
        results.overlap_length = len(data_ref_aligned) * results.pixel_size_ref

    else:
        # Partial profile comparison
        results.is_partial_profile = True

        # Determine which is longer (use equalized lengths)
        if len_ref_eq > len_comp_eq:
            trans_array, best_pos, corr, ref_aligned, comp_aligned = (
                align_partial_profile(prof_ref_eq, prof_comp_eq, params)
            )
            results.partial_profile_start = best_pos
        else:
            trans_array, best_pos, corr, comp_aligned, ref_aligned = (
                align_partial_profile(prof_comp_eq, prof_ref_eq, params)
            )
            results.partial_profile_start = best_pos

        results.transformation_array = trans_array
        results.correlation_coefficient = corr

        # Compute transformation
        cumulative_trans = np.eye(3)
        for i in range(trans_array.shape[0]):
            t = trans_array[i, 0]
            s = trans_array[i, 1]
            trans_mat = np.array([[s, 0, t], [0, 1, 0], [0, 0, 1]])
            cumulative_trans = trans_mat @ cumulative_trans

        results.position_shift = cumulative_trans[0, 2] * results.pixel_size_ref
        results.scale_factor = cumulative_trans[0, 0]
        if best_pos == 0.0:
            results.overlap_length = (
                min(len(ref_aligned), len(comp_aligned)) * results.pixel_size_ref
            )
        else:
            # This is the formula that matches the Matlab code. PV does not understand the '+1', if not use anymore, if statement is not necessary.
            results.overlap_length = (
                min(len(ref_aligned), len(comp_aligned)) + 1
            ) * results.pixel_size_ref - 0.5 * best_pos

        data_ref_aligned = ref_aligned
        data_comp_aligned = comp_aligned

    # Compute overlap ratio using EQUALIZED profile lengths
    # This is critical: pOverlap should be relative to the equalized lengths, not originals
    if len_ref_eq >= len_comp_eq:
        results.overlap_ratio = results.overlap_length / (
            len_comp_eq * results.pixel_size_comp
        )
    else:
        results.overlap_ratio = results.overlap_length / (
            len_ref_eq * results.pixel_size_ref
        )

    # Compute topographic measurements (convert to micrometers)
    p1_um = data_ref_aligned.flatten() * 1e6
    p2_um = data_comp_aligned.flatten() * 1e6
    p_diff_um = p2_um - p1_um

    N = len(p1_um)

    results.sa_ref = np.sum(np.abs(p1_um)) / N
    results.sq_ref = np.sqrt(np.dot(p1_um, p1_um) / N)
    results.sa_comp = np.sum(np.abs(p2_um)) / N
    results.sq_comp = np.sqrt(np.dot(p2_um, p2_um) / N)
    results.sa_diff = np.sum(np.abs(p_diff_um)) / N
    results.sq_diff = np.sqrt(np.dot(p_diff_um, p_diff_um) / N)

    # Signature differences
    if results.sq_ref > 0:
        results.ds1 = (results.sq_diff / results.sq_ref) ** 2
    if results.sq_comp > 0:
        results.ds2 = (results.sq_diff / results.sq_comp) ** 2
    if results.sq_ref > 0 and results.sq_comp > 0:
        results.ds = results.sq_diff**2 / (results.sq_ref * results.sq_comp)

    return results


# ============================================================================
# LEGACY COMPATIBILITY (for old interface)
# ============================================================================


def profile_correlator_single(
    profile_ref: Profile,
    profile_comp: Profile,
    results_table=None,
    param=None,
    iVerbose=0,
):
    """
    Legacy interface for MATLAB compatibility.

    This function maintains backward compatibility with the original MATLAB interface.
    """
    # Convert old-style params to new AlignmentParameters
    if param is None:
        params = AlignmentParameters()
    elif isinstance(param, dict):
        # Convert dict to AlignmentParameters
        params = AlignmentParameters(
            scale_passes=tuple(
                param.get("pass", [1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6])
            ),
            max_translation=param.get("max_translation", 10.0),
            max_scaling=param.get("max_scaling", 0.05),
            cutoff_hi=param.get("cutoff_hi", 1e-3),
            cutoff_lo=param.get("cutoff_lo", 5e-6),
            partial_mark_threshold=param.get("part_mark_perc", 8.0),
            use_mean=param.get("use_mean", True),
            remove_boundary_zeros=param.get("remove_zeros", True),
            show_info=param.get("show_info", False) or iVerbose > 0,
        )
    else:
        params = param

    # Run correlation
    results = correlate_profiles(profile_ref, profile_comp, params)

    # Convert to old-style results dict
    results_dict = {
        "bProfile": 1 if results.is_profile_comparison else 0,
        "bSegments": 0,
        "bPartialProfile": 1 if results.is_partial_profile else 0,
        "vPixSep1": results.pixel_size_ref,
        "vPixSep2": results.pixel_size_comp,
        "dPos": results.position_shift,
        "dScale": results.scale_factor,
        "startPartProfile": results.partial_profile_start
        if results.partial_profile_start is not None
        else np.nan,
        "lOverlap": results.overlap_length,
        "pOverlap": results.overlap_ratio,
        "ccf": results.correlation_coefficient,
        "simVal": results.correlation_coefficient,
        "metric": np.nan,
        "sa_1": results.sa_ref,
        "sq_1": results.sq_ref,
        "sa_2": results.sa_comp,
        "sq_2": results.sq_comp,
        "sa12": results.sa_diff,
        "sq12": results.sq_diff,
        "ds1": results.ds1,
        "ds2": results.ds2,
        "ds": results.ds,
        "pathReference": "",
        "pathCompare": "",
        "bKM": -1,
    }

    return results_dict
