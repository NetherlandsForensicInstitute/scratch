from dataclasses import dataclass, field
import numpy as np

from container_models.base import FloatArray1D, DepthData


@dataclass
class SurfaceMap:
    """
    Represents a 3D surface topography map with spatial metadata.

    :param height_map: The processed surface height data used for cell correlation.
    :param pixel_spacing: Pixel spacing [dx, dy] in micrometers, shape (2,).
    :param global_center: Image center [x, y] in micrometers, shape (2,).
    """

    height_map: DepthData
    pixel_spacing: FloatArray1D
    global_center: FloatArray1D

    @property
    def physical_size(self) -> FloatArray1D:
        """Physical dimensions [width, height] of the image in micrometers."""
        return np.flip(self.height_map.shape) * self.pixel_spacing


@dataclass
class CellResult:
    """
    Registration and similarity results for a single cell.

    :param center_reference: Cell center [x, y] on the reference surface (µm), shape (2,).
    :param center_comparison: Cell center [x, y] on the comparison surface (µm), shape (2,).
    :param registration_angle: Optimal rotation for this cell (radians).
    :param area_cross_correlation_function_score: Per-cell normalised cross-correlation
        coefficient (ACCF) at the best registration pose.
    :param reference_fill_fraction: Fraction of valid (non-NaN) pixels in the reference cell.
    :param is_congruent: True if this cell is classified as a Congruent Matching Cell (CMC).
    """

    center_reference: FloatArray1D
    center_comparison: FloatArray1D
    registration_angle: float
    area_cross_correlation_function_score: float
    reference_fill_fraction: float
    is_congruent: bool = False


@dataclass
class ComparisonResult:
    """
    Consolidated results of the CMC pipeline.

    :param cells: Per-cell registration and classification results.
    :param congruent_matching_cells_count: Number of cells classified as CMC.
    :param consensus_rotation: Rotation consensus across CMC cells (radians).
    :param consensus_translation: Translation consensus across CMC cells (µm), shape (2,).
    """

    cells: list[CellResult] = field(default_factory=list)
    congruent_matching_cells_count: int = 0
    consensus_rotation: float = 0.0
    consensus_translation: FloatArray1D = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )

    @property
    def cell_count(self) -> int:
        """Total number of cells."""
        return len(self.cells)

    @property
    def cmc_fraction(self) -> float:
        """Fraction of cells classified as CMC."""
        return (
            self.congruent_matching_cells_count / self.cell_count
            if self.cells
            else float("nan")
        )

    @property
    def cmc_area_fraction(self) -> float:
        """Fraction of valid surface area covered by CMC cells.

        Computed as the sum of reference_fill_fraction over congruent cells
        divided by the sum over all cells, weighted by actual non-NaN pixel counts.
        """
        total_area = sum(cell.reference_fill_fraction for cell in self.cells)
        if total_area == 0:
            return float("nan")
        cmc_area = sum(
            cell.reference_fill_fraction for cell in self.cells if cell.is_congruent
        )
        return cmc_area / total_area

    def update_summary(self) -> None:
        """Recount CMC cells from current cell statuses."""
        self.congruent_matching_cells_count = sum(
            1 for cell in self.cells if cell.is_congruent
        )


@dataclass
class ComparisonParams:
    """
    Parameters for the Congruent Matching Cells (CMC) algorithm.

    :param cell_size: Nominal cell size [width, height] in micrometers, shape (2,).
    :param minimum_fill_fraction: Minimum fraction of valid pixels required in a
        reference cell for it to be processed.
    :param correlation_threshold: Minimum per-cell ACCF score for CMC classification.
    :param angle_threshold: Maximum angular deviation from consensus for CMC (degrees).
    :param position_threshold: Maximum positional deviation from consensus for CMC (µm).
    :param search_angle_min: Lower bound of rotation search range (degrees).
    :param search_angle_max: Upper bound of rotation search range (degrees).
    :param search_angle_step: Angular step size for the coarse rotation sweep (degrees).
    """

    cell_size: FloatArray1D = field(
        default_factory=lambda: np.array([1000.0, 1000.0], dtype=np.float64)
    )
    minimum_fill_fraction: float = 0.5
    correlation_threshold: float = 0.4
    angle_threshold: float = 2.0
    position_threshold: float = 100.0
    search_angle_min: float = -180.0
    search_angle_max: float = 180.0
    search_angle_step: float = 1.0
