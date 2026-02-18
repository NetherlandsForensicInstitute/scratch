from dataclasses import dataclass, field
import numpy as np

from container_models.base import FloatArray1D, DepthData


@dataclass
class SurfaceMap:
    """
    Represents a 3D surface topography map with spatial metadata.

    :param height_map: The 'filtered' surface data used for correlation.
    :param pixel_spacing: spacing [dx, dy] in micrometers, shape (2,).
    :param global_center: center [x, y] in micrometers, shape (2,).
    :param orientation_angle: orientation in radians.
    :param unfiltered_height_map: The 'leveled' but unfiltered data used for global alignment and roughness metrics.
    """

    height_map: DepthData
    pixel_spacing: FloatArray1D
    global_center: FloatArray1D
    orientation_angle: float = 0.0
    unfiltered_height_map: DepthData | None = None

    def get_alignment_data(self) -> DepthData:
        """Returns unfiltered data if available, otherwise the primary height map."""
        return (
            self.unfiltered_height_map
            if self.unfiltered_height_map is not None
            else self.height_map
        )

    @property
    def physical_size(self) -> FloatArray1D:
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

    center_reference: FloatArray1D
    center_comparison: FloatArray1D
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
    consensus_translation: FloatArray1D = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    area_similarity: AreaSimilarityResult | None = None

    def update_summary(self) -> None:
        """Update count based on current cell statuses."""
        self.congruent_matching_cells_count = sum(
            1 for cell in self.cells if cell.is_congruent
        )


@dataclass
class ComparisonParams:
    """
    Settings for the Congruent Matching Cells (CMC) algorithm logic.

    :param cell_size: nominal size [width, height] in micrometers, shape (2,).
    :param minimum_fill_fraction: minimum valid data required to process a cell.
    :param correlation_threshold: minimum ACCF for a cell to be considered.
    :param angle_threshold: maximum allowed angular deviation (degrees).
    :param position_threshold: maximum allowed positional deviation (micrometers).
    :param search_angle_min: minimum angle to test during rotation search (degrees).
    :param search_angle_max: maximum angle to test during rotation search (degrees).
    :param search_angle_step: increment for the rotation search (degrees).
    """

    cell_size: FloatArray1D = field(
        default_factory=lambda: np.array([1000.0, 1000.0], dtype=np.float64)
    )
    minimum_fill_fraction: float = 0.5
    correlation_threshold: float = 0.4
    angle_threshold: float = 2.0
    position_threshold: float = 100.0
    search_angle_min: float = -5.0
    search_angle_max: float = 5.0
    search_angle_step: float = 0.5
