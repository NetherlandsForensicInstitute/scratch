"""
Profile correlator module for striated mark comparison.

This module provides functions for comparing 1D profiles of striated marks
(scratch marks, toolmarks, etc.) using global brute-force search and
correlation analysis.

The main entry point is :func:`correlate_profiles`, which handles the complete
comparison workflow including:

- Sampling distance equalization
- Global brute-force search over all shift positions and scale factors
- Selection of alignment with maximum cross-correlation
- Computation of comparison metrics

Submodules
----------
- data_types: Core data structures (Profile, AlignmentParameters, StriationComparisonResults)
- profile_correlator: Main entry point function (global brute-force search)
- transforms: Resampling operations for pixel scale equalization
- statistics: Statistical metrics (correlation, roughness, overlap ratio)
"""

# Core data types
from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    StriationComparisonResults,
    Profile,
)

# Main entry point
from conversion.profile_correlator.profile_correlator import correlate_profiles

# Mark-level wrapper
from conversion.profile_correlator.mark_correlator import (
    correlate_striation_marks,
    MarkCorrelationResult,
)

# Transform functions
from conversion.profile_correlator.transforms import equalize_pixel_scale

# Statistics functions
from conversion.profile_correlator.statistics import (
    compute_cross_correlation,
    compute_overlap_ratio,
    compute_roughness_sa,
    compute_roughness_sq,
    compute_normalized_square_based_roughness_differences,
)

# Re-export cutoff_to_gaussian_sigma from conversion.filter for convenience
from conversion.filter.gaussian import cutoff_to_gaussian_sigma

__all__ = [
    # Main entry point
    "correlate_profiles",
    # Mark-level wrapper
    "correlate_striation_marks",
    "MarkCorrelationResult",
    # Data types
    "Profile",
    "AlignmentParameters",
    "StriationComparisonResults",
    # Transforms
    "equalize_pixel_scale",
    # Statistics
    "compute_cross_correlation",
    "compute_overlap_ratio",
    "compute_roughness_sa",
    "compute_roughness_sq",
    "compute_normalized_square_based_roughness_differences",
    # Filtering (re-exported from conversion.filter)
    "cutoff_to_gaussian_sigma",
]
