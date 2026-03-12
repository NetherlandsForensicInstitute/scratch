from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.coarse import match_cells
from conversion.surface_comparison.grid import GridCell
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
    ProcessedMark,
)


def coarse_registration(
    grid_cells: list[GridCell],
    comparison_image: ScanImage,
    params: ComparisonParams,
) -> list[Cell]:
    """
    Register each reference grid cell against the comparison image.

    Computes the global mean of the reference image as a NaN fill value, then delegates to :func:`match_cells`
    to find the best-matching position and angle for every cell via a coarse angle sweep.

    :param grid_cells: Reference grid cells to register.
    :param comparison_image: Comparison scan image to search over.
    :param params: Algorithm parameters controlling the angle sweep and fill-fraction thresholds.
    :returns: List of :class:`Cell` objects with the best registration result per grid cell.
    """
    # TODO: Merge this with `match.py` when deprecating the MATLAB implementation.
    matched_cells = match_cells(
        grid_cells=grid_cells, comparison_image=comparison_image, params=params
    )
    return matched_cells


def fine_registration(comparison_mark: ProcessedMark, cells: list[Cell]) -> list[Cell]:
    """TODO: Implement this function."""
    return cells
