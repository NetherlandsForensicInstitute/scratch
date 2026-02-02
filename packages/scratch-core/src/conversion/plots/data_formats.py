from dataclasses import dataclass

from conversion.container_models.base import ImageRGB


@dataclass
class CorrelationMetrics:
    """
    Metrics from profile correlation comparison for display.

    :param score: Correlation coefficient.
    :param shift: Shift in µm.
    :param overlap: Overlap percentage.
    :param sq_a: Sq (RMS roughness) of reference surface A in µm.
    :param sq_b: Sq (RMS roughness) of compared surface B in µm.
    :param sq_b_minus_a: Sq of difference (B-A) in µm.
    :param sq_ratio: Sq(B) / Sq(A) percentage.
    :param sign_diff_dsab: Signed difference DsAB percentage.
    :param data_spacing: Data spacing in µm.
    :param quality_passbands: Mapping of (low, high) µm cutoffs to correlation values.
    """

    score: float
    shift: float
    overlap: float
    sq_a: float
    sq_b: float
    sq_b_minus_a: float
    sq_ratio: float
    sign_diff_dsab: float
    data_spacing: float
    quality_passbands: dict[tuple[float, float], float]


@dataclass
class StriationComparisonPlots:
    """
    Results from striation (profile) comparison visualization.

    :param similarity_plot: Aligned profiles overlaid.
    :param comparison_overview: Main results overview figure.
    :param mark1_filtered_preview_image: Filtered reference mark preview.
    :param mark2_filtered_preview_image: Filtered compared mark preview.
    :param mark1_vs_moved_mark2: Both marks side by side.
    :param wavelength_plot: Profiles + wavelength-dependent cross-correlation.
    """

    similarity_plot: ImageRGB
    comparison_overview: ImageRGB
    mark1_filtered_preview_image: ImageRGB
    mark2_filtered_preview_image: ImageRGB
    mark1_vs_moved_mark2: ImageRGB
    wavelength_plot: ImageRGB
