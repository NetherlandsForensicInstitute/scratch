from dataclasses import dataclass

from container_models.base import FloatArray1D, FloatArray2D, ImageRGB


@dataclass
class StriationComparisonMetrics:
    """
    Metrics from profile correlation comparison for display.

    :param correlation_coefficient: Pearson cross-correlation coefficient.
    :param position_shift: Shift in µm.
    :param overlap_ratio: Overlap percentage.
    :param mean_square_ref: Mean square roughness (Sq) of reference surface in µm.
    :param mean_square_comp: Mean square roughness (Sq) of compared surface in µm.
    :param mean_square_of_difference: Mean square roughness (Sq) of difference (comp - ref) in µm.
    :param mean_square_ratio: Sq(comp) / Sq(ref) percentage.
    :param signed_roughness_difference: Signed roughness difference percentage.
    :param pixel_size: Data spacing in µm.
    :param quality_passbands: Mapping of (low, high) µm cutoffs to correlation values.
    """

    correlation_coefficient: float
    position_shift: float
    overlap_ratio: float
    mean_square_ref: float
    mean_square_comp: float
    mean_square_of_difference: float
    mean_square_ratio: float
    signed_roughness_difference: float
    pixel_size: float
    quality_passbands: dict[tuple[float, float], float]


@dataclass
class StriationComparisonPlots:
    """
    Results from striation (profile) comparison visualization.

    :param similarity_plot: Aligned profiles overlaid.
    :param comparison_overview: Main results overview figure.
    :param filtered_reference_heatmap: Filtered reference preview image.
    :param filtered_compared_heatmap: Filtered compared preview image.
    :param side_by_side_heatmap: Both marks side by side preview image.
    :param wavelength_plot: Profiles + wavelength-dependent cross-correlation.
    """

    similarity_plot: ImageRGB
    comparison_overview: ImageRGB
    filtered_reference_heatmap: ImageRGB
    filtered_compared_heatmap: ImageRGB
    side_by_side_heatmap: ImageRGB
    wavelength_plot: ImageRGB


@dataclass
class ImpressionComparisonMetrics:
    """
    Metrics for impression comparison display.

    :param area_correlation: Areal correlation coefficient (from area-based comparison).
    :param cell_correlations: Grid of per-cell correlation values (shape: n_rows x n_cols).
    :param cmc_score: Congruent Matching Cells score (percentage of cells above threshold).
    :param mean_square_ref: Mean square roughness (Sq) of reference surface in µm.
    :param mean_square_comp: Mean square roughness (Sq) of compared surface in µm.
    :param mean_square_of_difference: Mean square roughness (Sq) of difference (comp - ref) in µm.
    :param has_area_results: Whether area-based results were computed.
    :param has_cell_results: Whether cell/CMC-based results were computed.
    :param cell_positions_compared: (n_cells, 2) matched positions on compared surface in µm, row-major order.
    :param cell_rotations_compared: (n_cells,) rotation angles in radians, row-major order.
    :param cmc_area_fraction: Percentage of total surface area covered by CMC cells.
    :param cutoff_low_pass: Cutoff length low-pass filter in µm.
    :param cutoff_high_pass: Cutoff length high-pass filter in µm.
    :param cell_size_um: Cell size in µm.
    :param max_error_cell_position: Max error cell position in µm.
    :param max_error_cell_angle: Max error cell angle in degrees.
    :param cell_similarity_threshold: Minimum correlation for a cell to be CMC (default 0.25).
    """

    area_correlation: float
    cell_correlations: FloatArray2D
    cmc_score: float
    mean_square_ref: float
    mean_square_comp: float
    mean_square_of_difference: float
    has_area_results: bool
    has_cell_results: bool
    cell_positions_compared: FloatArray2D
    cell_rotations_compared: FloatArray1D
    cmc_area_fraction: float
    cutoff_low_pass: float
    cutoff_high_pass: float
    cell_size_um: float
    max_error_cell_position: float
    max_error_cell_angle: float
    cell_similarity_threshold: float = 0.25


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
