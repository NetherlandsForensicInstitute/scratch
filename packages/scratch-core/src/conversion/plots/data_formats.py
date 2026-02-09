from dataclasses import dataclass

from container_models.base import FloatArray1D, FloatArray2D, ImageRGB


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


@dataclass
class ImpressionComparisonMetrics:
    """
    Metrics for impression comparison display.

    :param area_correlation: Areal correlation coefficient (from area-based comparison).
    :param cell_correlations: Grid of per-cell correlation values (shape: n_rows x n_cols).
    :param cmc_score: Congruent Matching Cells score (percentage of cells above threshold).
    :param sq_ref: Sq (RMS roughness) of reference surface in µm.
    :param sq_comp: Sq (RMS roughness) of compared surface in µm.
    :param sq_diff: Sq of difference (comp - ref) in µm.
    :param has_area_results: Whether area-based results were computed.
    :param has_cell_results: Whether cell/CMC-based results were computed.
    :param cell_positions_compared: (n_cells, 2) matched positions on compared surface in µm, row-major order.
    :param cell_rotations_compared: (n_cells,) rotation angles in radians, row-major order.
    :param cell_similarity_threshold: Minimum correlation for a cell to be CMC (default 0.25).
    """

    area_correlation: float
    cell_correlations: FloatArray2D
    cmc_score: float
    sq_ref: float
    sq_comp: float
    sq_diff: float
    has_area_results: bool
    has_cell_results: bool
    cell_positions_compared: FloatArray2D | None = None
    cell_rotations_compared: FloatArray1D | None = None
    cell_similarity_threshold: float = 0.25


@dataclass
class ImpressionComparisonPlots:
    """
    Results from impression mark comparison visualization.

    Contains rendered images for both area-based and cell/CMC-based visualizations.
    Fields are None when the corresponding analysis was not performed.

    :param comparison_overview: Combined overview figure with all results.
    :param leveled_reference: Leveled reference surface visualization.
    :param leveled_compared: Leveled compared surface visualization.
    :param filtered_reference: Filtered reference surface visualization.
    :param filtered_compared: Filtered compared surface visualization.
    :param cell_reference: Cell-preprocessed reference visualization.
    :param cell_compared: Cell-preprocessed compared visualization.
    :param cell_overlay: All cells overlay visualization.
    :param cell_cross_correlation: Cell-based cross-correlation heatmap.
    """

    comparison_overview: ImageRGB
    # Area-based plots
    leveled_reference: ImageRGB | None
    leveled_compared: ImageRGB | None
    filtered_reference: ImageRGB | None
    filtered_compared: ImageRGB | None
    # Cell/CMC-based plots
    cell_reference: ImageRGB | None
    cell_compared: ImageRGB | None
    cell_overlay: ImageRGB | None
    cell_cross_correlation: ImageRGB | None
