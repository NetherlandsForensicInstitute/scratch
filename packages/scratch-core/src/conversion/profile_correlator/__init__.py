from .profile_correlator import (
    # Main data structures
    Profile,
    AlignmentParameters,
    ComparisonResults,
    # Main correlation function
    correlate_profiles,
    # Legacy compatibility
    profile_correlator_single,
    # Utility functions (for advanced use)
    compute_similarity_score,
    apply_transformation,
    apply_multi_scale_transformation,
    equalize_pixel_sizes,
    make_equal_length,
    remove_boundary_zeros,
    # Alignment functions (for advanced use)
    align_profiles_multiscale,
    align_partial_profile,
)

# Define public API
__all__ = [
    # Primary API (recommended)
    "Profile",
    "AlignmentParameters",
    "ComparisonResults",
    "correlate_profiles",
    # Legacy compatibility
    "profile_correlator_single",
    # Utility functions
    "compute_similarity_score",
    "apply_transformation",
    "apply_multi_scale_transformation",
    "equalize_pixel_sizes",
    "make_equal_length",
    "remove_boundary_zeros",
    # Advanced alignment
    "align_profiles_multiscale",
    "align_partial_profile",
]
