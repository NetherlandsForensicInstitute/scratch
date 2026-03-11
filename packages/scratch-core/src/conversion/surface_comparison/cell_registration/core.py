from collections.abc import Iterable

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.match import match_cells
from conversion.surface_comparison.grid import GridCell
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
    ProcessedMark,
)
import numpy as np


def coarse_registration(
    grid_cells: Iterable[GridCell],
    reference_image: ScanImage,
    comparison_image: ScanImage,
    params: ComparisonParams,
) -> list[Cell]:
    """TODO: Write docstring."""
    fill_value_reference = float(np.nanmean(reference_image.data))
    matched_cells = match_cells(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
        fill_value_reference=fill_value_reference,
    )
    return matched_cells


def fine_registration(
    comparison_mark: ProcessedMark, cells: Iterable[Cell]
) -> list[Cell]:
    """TODO: Implement function."""
    return list(cells)
