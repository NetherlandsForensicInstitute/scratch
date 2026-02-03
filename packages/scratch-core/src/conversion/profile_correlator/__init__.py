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

The module structure follows the patterns established in the preprocess_impression
module, using dataclasses for parameters and results.

Submodules
----------
- data_types: Core data structures (Profile, AlignmentParameters, etc.)
- correlator: Main entry point function (global brute-force search)
- transforms: Translation, scaling, and resampling operations
- similarity: Cross-correlation and comparison metrics
"""

# Core data types
from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    ComparisonResults,
    Profile,
    TransformParameters,
)

# Main entry point
from conversion.profile_correlator.correlator import correlate_profiles

# Transform functions
from conversion.profile_correlator.transforms import (
    apply_transform,
    compute_cumulative_transform,
    equalize_pixel_scale,
    make_profiles_equal_length,
)

# Similarity functions
from conversion.profile_correlator.similarity import (
    compute_comparison_metrics,
    compute_cross_correlation,
)

# Re-export cutoff_to_gaussian_sigma from conversion.filter for convenience
from conversion.filter.gaussian import cutoff_to_gaussian_sigma

__all__ = [
    # Main entry point
    "correlate_profiles",
    # Data types
    "Profile",
    "AlignmentParameters",
    "ComparisonResults",
    "TransformParameters",
    # Transforms
    "equalize_pixel_scale",
    "make_profiles_equal_length",
    "apply_transform",
    "compute_cumulative_transform",
    # Similarity
    "compute_cross_correlation",
    "compute_comparison_metrics",
    # Filtering (re-exported from conversion.filter)
    "cutoff_to_gaussian_sigma",
]
