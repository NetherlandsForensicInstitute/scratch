from dataclasses import dataclass

import numpy as np


@dataclass
class CorrelationMetrics:
    """Metrics from profile correlation comparison for display."""

    score: float  # Correlation coefficient
    shift: float  # Shift in µm
    overlap: float  # Overlap percentage
    sq_a: float  # Sq (RMS roughness) of reference surface A in µm
    sq_b: float  # Sq (RMS roughness) of compared surface B in µm
    sq_b_minus_a: float  # Sq of difference (B-A) in µm
    sq_ratio: float  # Sq(B) / Sq(A) percentage
    sign_diff_dsab: float  # Signed difference DsAB percentage
    data_spacing: float  # Data spacing in µm
    quality_passbands: dict[tuple[float, float], float]  # (low, high) µm → correlation


@dataclass
class StriationComparisonPlots:
    """Results from striation (profile) comparison visualization."""

    # Comparison plots
    similarity_plot: np.ndarray  # Aligned profiles overlaid
    comparison_overview: np.ndarray  # Main NFI results overview figure

    # Filtered mark images
    mark1_filtered_preview_image: np.ndarray  # Filtered reference mark preview
    mark2_filtered_preview_image: np.ndarray  # Filtered compared mark preview

    # Side by side and wavelength plots
    mark1_vs_moved_mark2: np.ndarray  # Both marks side by side with gap
    wavelength_plot: np.ndarray  # Profiles + wavelength-dependent cross-correlation
