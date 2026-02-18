from dataclasses import dataclass, field
import numpy as np


@dataclass
class SurfaceMap:
    """
    Represents a 3D surface topography map with spatial metadata.

    :param height_map: 2D array of height values in micrometers, shape (rows, columns).
    :param pixel_spacing: spacing [dx, dy] in micrometers, shape (2,).
    :param global_center: center [x, y] in micrometers, shape (2,).
    :param orientation_angle: orientation in radians.
    :param unfiltered_height_map: leveled version of the height map, shape (rows, columns).
    """

    height_map: np.ndarray
    pixel_spacing: np.ndarray
    global_center: np.ndarray
    orientation_angle: float = 0.0
    unfiltered_height_map: np.ndarray | None = None

    @property
    def physical_size(self) -> np.ndarray:
        """
        Total width and height [x, y] in micrometers.

        :returns: physical size array, shape (2,).
        """
        return np.flip(self.height_map.shape) * self.pixel_spacing


@dataclass
class CellResult:
    """
    Registration and similarity results for a single cell.

    :param center_reference: center [x, y] on reference surface (micrometers), shape (2,).
    :param center_comparison: center [x, y] on comparison surface (micrometers), shape (2,).
    :param registration_angle: optimal rotation found for this cell (radians).
    :param area_cross_correlation_function_score: The maximum normalized cross-correlation coefficient.
        In NIST literature, this is the Area Cross-Correlation Function (ACCF) value.
    :param reference_fill_fraction: fraction of valid pixels in the reference cell.
    :param is_congruent: flag indicating if the cell is classified as a Congruent Matching Cell (CMC).
    """

    center_reference: np.ndarray
    center_comparison: np.ndarray
    registration_angle: float
    area_cross_correlation_function_score: float
    reference_fill_fraction: float
    is_congruent: bool = False


@dataclass
class AreaSimilarityResult:
    """
    Global similarity metrics between two surfaces.

    :param cross_correlation_coefficient: global similarity value.
    :param overlap_fraction: fraction of overlapping valid area.
    :param reference_root_mean_square_roughness: The Sq parameter (RMS height) for the reference.
    :param comparison_root_mean_square_roughness: The Sq parameter (RMS height) for the comparison.
    """

    cross_correlation_coefficient: float
    overlap_fraction: float
    reference_root_mean_square_roughness: float
    comparison_root_mean_square_roughness: float


@dataclass
class ComparisonResult:
    """
    Consolidated results of the full comparison pipeline.

    :param cells: list of individual cell registration results.
    :param congruent_matching_cells_count: total count of congruent matching cells (CMCs) found.
    :param consensus_rotation: the global rotation consensus (radians).
    :param consensus_translation: global translation consensus (micrometers), shape (2,).
    """

    cells: list[CellResult] = field(default_factory=list)
    congruent_matching_cells_count: int = 0
    consensus_rotation: float = 0.0
    consensus_translation: np.ndarray = field(default_factory=lambda: np.zeros(2))
    area_similarity: AreaSimilarityResult | None = None

    def update_summary(self) -> None:
        """Update count based on current cell statuses."""
        self.congruent_matching_cells_count = sum(
            1 for cell in self.cells if cell.is_congruent
        )
