from dataclasses import dataclass

from container_models.base import FloatArray1D, ImageRGB


@dataclass
class DensityData:
    """
    Kernel density estimates for KM and KNM score distributions.

    :param x: x values at which densities are evaluated.
    :param km_density_at_x: KM density values at x.
    :param knm_density_at_x: KNM density values at x.
    """

    x: FloatArray1D
    km_density_at_x: FloatArray1D
    knm_density_at_x: FloatArray1D


@dataclass
class HistogramData:
    """
    Input data for score histogram plots.

    :param scores: Array of score values.
    :param labels: Array of labels (0 for KNM, 1 for KM).
    :param bins: Number of bins for histogram. If None, uses 'auto' binning.
    :param densities: Optional density curves for histogram overlay.
    :param new_score: Optional score value to display as a vertical line on the histogram.
    """

    scores: FloatArray1D
    labels: FloatArray1D
    bins: int | None = None
    densities: DensityData | None = None
    new_score: float | None = None


@dataclass
class LlrTransformationData:
    """
    Input data for Log10LR transformation plots.

    :param scores: Score axis values.
    :param llrs: Log10LR values.
    :param llrs_at5: Log10LR values at 5% confidence.
    :param llrs_at95: Log10LR values at 95% confidence.
    :param score_llr_point: Optional (score, llr) coordinate to mark the score on the LLR transformation plot.
    """

    scores: FloatArray1D
    llrs: FloatArray1D
    llrs_at5: FloatArray1D
    llrs_at95: FloatArray1D
    score_llr_point: tuple[float, float] | None


@dataclass
class StriationComparisonPlots:
    """
    Results from striation (profile) comparison visualization.

    :param similarity_plot: Aligned profiles overlaid.
    :param comparison_overview: Main results overview figure.
    :param filtered_reference_heatmap: Filtered reference preview image.
    :param filtered_compared_heatmap: Filtered compared preview image.
    :param side_by_side_heatmap: Both marks side by side preview image.
    """

    similarity_plot: ImageRGB
    comparison_overview: ImageRGB
    filtered_reference_heatmap: ImageRGB
    filtered_compared_heatmap: ImageRGB
    side_by_side_heatmap: ImageRGB


@dataclass
class ImpressionComparisonPlots:
    """
    Results from impression mark comparison visualization.

    Contains rendered images for both area-based and cell/CMC-based visualizations.
    Fields are None when the corresponding analysis was not performed.

    :param comparison_overview: Combined overview figure with all results.
    :param leveled_reference_heatmap: Leveled reference preview image.
    :param leveled_compared_heatmap: Leveled compared preview image.
    :param filtered_reference_heatmap: Filtered reference preview image.
    :param filtered_compared_heatmap: Filtered compared preview image.
    :param cell_reference_heatmap: Cell-preprocessed reference preview image.
    :param cell_compared_heatmap: Cell-preprocessed compared preview image.
    :param cell_overlay: All cells overlay visualization.
    :param cell_cross_correlation: Cell-based cross-correlation heatmap.
    """

    comparison_overview: ImageRGB
    leveled_reference_heatmap: ImageRGB
    leveled_compared_heatmap: ImageRGB
    filtered_reference_heatmap: ImageRGB
    filtered_compared_heatmap: ImageRGB
    cell_reference_heatmap: ImageRGB
    cell_compared_heatmap: ImageRGB
    cell_overlay: ImageRGB
    cell_cross_correlation: ImageRGB
