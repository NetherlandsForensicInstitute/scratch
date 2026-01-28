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
        translation: Translation in samples (positive = shift right)
        scaling: Scaling factor

    Returns:
        Transformed profile
    """
    n = len(data)
    x_orig = np.arange(n, dtype=float)

    # To shift data RIGHT by translation, we sample from LEFT (subtract translation)
    # x_query are the positions in the ORIGINAL data we need to sample from
    x_query = (x_orig - translation) / scaling

    # Interpolate: maps original indices -> data values
    interpolator = interp1d(
        x_orig, data, kind="linear", bounds_error=False, fill_value=0.0
    )

    return interpolator(x_query)


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
    Crop profiles to equal length by removing from the end.

    Note: We crop from the END to preserve alignment at the beginning,
    since profiles are typically aligned at their start positions.
    Center-cropping would introduce an artificial shift that the optimizer
    would need to compensate for.

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

    # Crop from end to preserve alignment at start
    return profile_1[:min_len], profile_2[:min_len]


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
    """Bounded optimization - matches MATLAB's fminsearchbnd behavior."""

    @staticmethod
    def optimize(
        func,
        x0: NDArray[np.floating],
        bounds: Tuple[Tuple[float, float], ...],
        args: tuple = (),
        global_search: bool = False,
    ) -> Tuple[NDArray[np.floating], float]:
        """
        Perform optimization.

        Args:
            func: Objective function
            x0: Initial guess
            bounds: Parameter bounds
            args: Additional arguments for func
            global_search: If True, do grid search first

        Returns:
            Tuple of (optimal_params, function_value)
        """
        if global_search:
            # Grid search for first scale to find approximate solution
            trans_min, trans_max = bounds[0]
            n_grid = 51
            trans_grid = np.linspace(trans_min, trans_max, n_grid)

            best_trans = 0.0
            best_error = float("inf")

            for t in trans_grid:
                try:
                    err = func(np.array([t, 0.0]), *args)
                    if err < best_error:
                        best_error = err
                        best_trans = t
                except Exception as _:
                    continue

            x0_start = np.array([best_trans, 0.0])
        else:
            x0_start = x0

        # Use Nelder-Mead with small initial simplex for local refinement
        n = len(x0_start)
        initial_simplex = np.zeros((n + 1, n))
        initial_simplex[0] = x0_start
        initial_simplex[1] = x0_start + np.array([0.5, 0.0])
        initial_simplex[2] = x0_start + np.array([0.0, 10.0])

        try:
            result = minimize(
                func,
                x0_start,
                args=args,
                method="Nelder-Mead",
                options={
                    "xatol": 1e-4,
                    "fatol": 1e-6,
                    "maxiter": 1000,
                    "initial_simplex": initial_simplex,
                },
            )
            optimal_params = result.x
        except Exception as _:
            # Fallback to starting point if optimization fails
            optimal_params = x0_start

        # Clip to bounds
        optimal_params = np.array(
            [
                np.clip(optimal_params[0], bounds[0][0], bounds[0][1]),
                np.clip(optimal_params[1], bounds[1][0], bounds[1][1]),
            ]
        )

        final_error = func(optimal_params, *args)
        return optimal_params, final_error


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
        # Constrain max_translation to profile length - MATLAB uses huge values that work
        # with fminsearchbnd but cause scipy optimizers to hit boundaries
        max_trans_from_params = params.max_translation / pixel_size
        max_trans_from_profile = len(data_ref)  # Can't shift more than profile length
        max_trans = min(max_trans_from_params, max_trans_from_profile)
        max_scale = params.max_scaling

        # MATLAB shrinks translation bounds based on cumulative translation
        # This prevents runaway translations across scales
        max_trans_positive = (
            max_trans - cumulative_translation
        )  # Remaining positive translation allowed
        max_trans_negative = (
            max_trans + cumulative_translation
        )  # Remaining negative translation allowed

        max_trans_samples_pos = max_trans_positive / subsample_factor
        max_trans_samples_neg = max_trans_negative / subsample_factor

        bounds = (
            (-max_trans_samples_neg, max_trans_samples_pos),
            (
                (-max_scale / cumulative_scaling) * 10000,
                (max_scale / cumulative_scaling) * 10000,
            ),
        )

        # Initial guess
        x0 = np.array([0.0, 0.0])

        # Optimize - use global search on first scale to find correct region
        is_first_scale = len(trans_array) == 0
        optimal_params, _ = BoundedOptimizer.optimize(
            error_function,
            x0,
            bounds,
            args=(subsampled_ref, subsampled_comp),
            global_search=is_first_scale,
        )

        # Extract transformation
        translation = optimal_params[0] * subsample_factor
        scaling = optimal_params[1] / 10000.0 + 1.0

        # Update cumulative transformation
        cumulative_translation += translation
        new_cumulative_scaling = cumulative_scaling * scaling

        # Enforce total cumulative scaling stays within max_scaling bounds
        # This prevents compounding small per-iteration changes from exceeding limits
        min_total_scale = 1.0 - params.max_scaling
        max_total_scale = 1.0 + params.max_scaling

        if new_cumulative_scaling < min_total_scale:
            # Clamp to minimum and adjust this iteration's scaling
            scaling = min_total_scale / cumulative_scaling
            new_cumulative_scaling = min_total_scale
        elif new_cumulative_scaling > max_total_scale:
            # Clamp to maximum and adjust this iteration's scaling
            scaling = max_total_scale / cumulative_scaling
            new_cumulative_scaling = max_total_scale

        cumulative_scaling = new_cumulative_scaling

        # Store
        trans_array.append([translation, scaling])

        # Apply ALL accumulated transformations to original data (like MATLAB's TranslateScalePointset)
        trans_array_np = np.array(trans_array)
        data_comp_transformed = apply_multi_scale_transformation(
            data_comp, trans_array_np
        )

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

        # Align this segment with tighter bounds since we already found approximate position
        # MATLAB uses fminsearchbnd which does local refinement from x0=[0,0]
        # We need to restrict translation to prevent runaway optimization
        params_tight = AlignmentParameters(
            scale_passes=params.scale_passes,
            max_translation=min(
                params.max_translation, len_partial * profile_ref.pixel_size * 0.1
            ),  # 10% of profile
            max_scaling=params.max_scaling,
            cutoff_hi=params.cutoff_hi,
            cutoff_lo=params.cutoff_lo,
            use_mean=params.use_mean,
            remove_boundary_zeros=params.remove_boundary_zeros,
            partial_mark_threshold=params.partial_mark_threshold,
        )

        try:
            trans, ref_aligned, partial_aligned, xcorr = align_profiles_multiscale(
                ref_segment, partial_temp, params_tight
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
        best_ref_segment if best_ref_segment is not None else data_ref[:len_partial],
        best_aligned if best_aligned is not None else data_partial,
    )


# ============================================================================
# VISUALIZATION
# ============================================================================


def _save_alignment_figure(
    figure_path: str,
    profile_ref: Profile,
    profile_comp: Profile,
    prof_ref_eq: Profile,
    prof_comp_eq: Profile,
    data_ref_aligned: NDArray[np.floating],
    data_comp_aligned: NDArray[np.floating],
    results: "ComparisonResults",
) -> None:
    """
    Save alignment visualization figure.

    Args:
        figure_path: Path to save the figure
        profile_ref: Original reference profile
        profile_comp: Original comparison profile
        prof_ref_eq: Equalized reference profile
        prof_comp_eq: Equalized comparison profile
        data_ref_aligned: Aligned reference data
        data_comp_aligned: Aligned comparison data
        results: Comparison results
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Get profile data
    data_ref = prof_ref_eq.mean_profile()
    data_comp = prof_comp_eq.mean_profile()
    pixel_size_um = prof_ref_eq.pixel_size * 1e6

    # Create x-axis in micrometers
    x_ref = np.arange(len(data_ref)) * pixel_size_um
    x_comp = np.arange(len(data_comp)) * pixel_size_um
    x_aligned = np.arange(len(data_ref_aligned)) * pixel_size_um

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f"Profile Alignment\n"
        f"pOverlap={results.overlap_ratio:.4f}, ccf={results.correlation_coefficient:.4f}, "
        f"partial={results.is_partial_profile}",
        fontsize=12,
    )

    # Plot 1: Original profiles (after equalization)
    ax1 = axes[0]
    ax1.plot(
        x_ref,
        data_ref * 1e6,
        "b-",
        label=f"Reference ({len(data_ref)} samples)",
        alpha=0.8,
    )
    ax1.plot(
        x_comp,
        data_comp * 1e6,
        "r-",
        label=f"Comparison ({len(data_comp)} samples)",
        alpha=0.8,
    )
    ax1.set_xlabel("Position (μm)")
    ax1.set_ylabel("Height (μm)")
    ax1.set_title("Original Profiles (after pixel size equalization)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Aligned profiles
    ax2 = axes[1]
    ax2.plot(
        x_aligned,
        data_ref_aligned.flatten() * 1e6,
        "b-",
        label="Reference (aligned)",
        alpha=0.8,
    )
    ax2.plot(
        x_aligned,
        data_comp_aligned.flatten() * 1e6,
        "r-",
        label="Comparison (aligned)",
        alpha=0.8,
    )
    ax2.set_xlabel("Position (μm)")
    ax2.set_ylabel("Height (μm)")
    ax2.set_title(
        f"Aligned Profiles (shift={results.position_shift * 1e6:.1f}μm, scale={results.scale_factor:.6f})"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Correlation scan (for partial profiles) or difference
    ax3 = axes[2]

    if len(data_ref) != len(data_comp):
        # Compute correlation at each position for partial profiles
        if len(data_ref) > len(data_comp):
            longer, shorter = data_ref, data_comp
        else:
            longer, shorter = data_comp, data_ref

        n_positions = len(longer) - len(shorter) + 1
        correlations = []
        positions = []

        for pos in range(n_positions):
            segment = longer[pos : pos + len(shorter)]
            corr = compute_similarity_score(segment, shorter)
            correlations.append(corr)
            positions.append(pos * pixel_size_um)

        ax3.plot(positions, correlations, "b-", linewidth=1, label="Raw correlation")

        # Mark max correlation position
        if correlations:
            max_idx = np.argmax(correlations)
            ax3.scatter(
                [positions[max_idx]],
                [correlations[max_idx]],
                color="purple",
                s=100,
                zorder=5,
                label=f"Max: {positions[max_idx]:.0f}μm ({correlations[max_idx]:.3f})",
            )

        # Mark found position
        if results.partial_profile_start is not None:
            ax3.axvline(
                results.partial_profile_start * 1e6,
                color="g",
                linestyle="--",
                linewidth=2,
                label=f"Found: {results.partial_profile_start * 1e6:.0f}μm",
            )

        ax3.set_xlabel("Start Position (μm)")
        ax3.set_ylabel("Correlation")
        ax3.set_title("Correlation vs Start Position")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.5, 1.0)
    else:
        # Show difference for equal-length profiles
        diff = (data_comp_aligned.flatten() - data_ref_aligned.flatten()) * 1e6
        ax3.plot(x_aligned, diff, "g-", linewidth=1)
        ax3.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax3.set_xlabel("Position (μm)")
        ax3.set_ylabel("Difference (μm)")
        ax3.set_title(f"Profile Difference (RMS={results.sq_diff:.4f}μm)")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN CORRELATION FUNCTION
# ============================================================================


def correlate_profiles(
    profile_ref: Profile,
    profile_comp: Profile,
    params: Optional[AlignmentParameters] = None,
    figure_path: Optional[str] = None,
) -> ComparisonResults:
    """
    Main function to correlate two profiles.

    Args:
        profile_ref: Reference profile
        profile_comp: Comparison profile
        params: Alignment parameters (uses defaults if None)
        figure_path: Optional path to save alignment visualization figure

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
        # MATLAB: results_table.lOverlap = length(profiles1)*results_table.vPixSep1
        # Simply use the length of the aligned profile
        results.overlap_length = len(ref_aligned) * results.pixel_size_ref

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

    # Generate visualization if requested
    if figure_path is not None:
        _save_alignment_figure(
            figure_path=figure_path,
            profile_ref=profile_ref,
            profile_comp=profile_comp,
            prof_ref_eq=prof_ref_eq,
            prof_comp_eq=prof_comp_eq,
            data_ref_aligned=data_ref_aligned,
            data_comp_aligned=data_comp_aligned,
            results=results,
        )

    return results
