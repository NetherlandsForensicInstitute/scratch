"""
Profile correlator module for striated mark comparison.

This module provides functions for comparing 1D profiles of striated marks
(scratch marks, toolmarks, etc.) using multi-scale registration and
correlation analysis.

The main entry point is :func:`correlate_profiles`, which handles the complete
comparison workflow including:

- Sampling distance equalization
- Length matching (full or partial profile comparison)
- Multi-scale coarse-to-fine alignment
- Computation of comparison metrics

The module structure follows the patterns established in the preprocess_impression
module, using dataclasses for parameters and results.

Submodules
----------
- data_types: Core data structures (Profile, AlignmentParameters, etc.)
- correlator: Main entry point function
- alignment: Multi-scale alignment algorithms
- transforms: Translation, scaling, and resampling operations
- similarity: Cross-correlation and comparison metrics
- filtering: 1D Gaussian filtering with NaN handling
- candidate_search: Partial profile brute-force candidate detection
"""

# Core data types
from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    AlignmentResult,
    ComparisonResults,
    Profile,
    TransformParameters,
)

# Main entry point
from conversion.profile_correlator.correlator import correlate_profiles

# Alignment functions
from conversion.profile_correlator.alignment import (
    align_partial_profile_multiscale,
    align_profiles_multiscale,
)

# Transform functions
from conversion.profile_correlator.transforms import (
    apply_transform,
    compute_cumulative_transform,
    equalize_pixel_scale,
    make_profiles_equal_length,
    remove_boundary_zeros,
)

# Similarity functions
from conversion.profile_correlator.similarity import (
    compute_comparison_metrics,
    compute_cross_correlation,
)

# Filtering functions
from conversion.profile_correlator.filtering import (
    CHEBY_TO_GAUSS_FACTOR,
    apply_highpass_filter_1d,
    apply_lowpass_filter_1d,
    convolve_with_nan_handling,
    cutoff_to_gaussian_sigma,
)

# Candidate search
from conversion.profile_correlator.candidate_search import find_match_candidates

__all__ = [
    # Main entry point
    "correlate_profiles",
    # Data types
    "Profile",
    "AlignmentParameters",
    "AlignmentResult",
    "ComparisonResults",
    "TransformParameters",
    # Alignment
    "align_profiles_multiscale",
    "align_partial_profile_multiscale",
    # Transforms
    "equalize_pixel_scale",
    "make_profiles_equal_length",
    "apply_transform",
    "remove_boundary_zeros",
    "compute_cumulative_transform",
    # Similarity
    "compute_cross_correlation",
    "compute_comparison_metrics",
    # Filtering
    "CHEBY_TO_GAUSS_FACTOR",
    "cutoff_to_gaussian_sigma",
    "apply_lowpass_filter_1d",
    "apply_highpass_filter_1d",
    "convolve_with_nan_handling",
    # Candidate search
    "find_match_candidates",
]
