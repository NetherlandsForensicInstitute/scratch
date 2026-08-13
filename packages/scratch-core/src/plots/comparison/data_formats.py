from dataclasses import dataclass

from container_models.base import ImageRGB


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
    :param raw_reference_heatmap: Raw reference preview image.
    :param raw_compared_heatmap: Raw compared preview image.
    :param filtered_reference_heatmap: Filtered reference preview image.
    :param filtered_compared_heatmap: Filtered compared preview image.
    :param cell_reference_heatmap: Cell-preprocessed reference preview image.
    :param cell_compared_heatmap: Cell-preprocessed compared preview image.
    :param cell_overlay: All cells overlay visualization.
    :param cell_cross_correlation: Cell-based cross-correlation heatmap.
    """

    comparison_overview: ImageRGB
    raw_reference_heatmap: ImageRGB
    raw_compared_heatmap: ImageRGB
    filtered_reference_heatmap: ImageRGB
    filtered_compared_heatmap: ImageRGB
    cell_reference_heatmap: ImageRGB
    cell_compared_heatmap: ImageRGB
    cell_overlay: ImageRGB
    cell_cross_correlation: ImageRGB
