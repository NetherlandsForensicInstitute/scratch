"""
Unified filtering module for impression and striation preprocessing.

This package provides Gaussian regression filtering and related operations for
surface texture analysis, following ISO 16610 standards.
"""

from conversion.filter.gaussian import (
    ALPHA_GAUSSIAN,
    ALPHA_REGRESSION,
    apply_gaussian_regression_filter,
    apply_striation_preserving_filter_1d,
    cutoff_to_gaussian_sigma,
    gaussian_sigma_to_cutoff,
)
from conversion.filter.mark_filters import (
    apply_filter_pipeline,
    apply_gaussian_filter_mark,
)

__all__ = [
    # Constants
    "ALPHA_GAUSSIAN",
    "ALPHA_REGRESSION",
    # Conversion functions
    "cutoff_to_gaussian_sigma",
    "gaussian_sigma_to_cutoff",
    # Core filters
    "apply_gaussian_regression_filter",
    "apply_striation_preserving_filter_1d",
    # Mark-specific filters
    "apply_gaussian_filter_mark",
    "apply_filter_pipeline",
]
