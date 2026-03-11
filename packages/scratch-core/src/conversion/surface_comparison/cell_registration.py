from collections.abc import Iterable

from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
    CellMetaData,
    GridCell,
    ProcessedMark,
)

from conversion.surface_comparison.utils import convert_pixels_to_meters


def coarse_registration(
    grid_cells: Iterable[GridCell],
    comparison_image: ScanImage,
    params: ComparisonParams,
    fill_value_reference: float,  # Fill value for NaNs in the grid cell data
) -> list[Cell]:
    """TODO: Implement function."""

    # Generate dummy output
    pixel_size = comparison_image.scale_x  # Assumes isotropic image
    output = []
    for grid_cell in grid_cells:
        dummy = Cell(
            center_reference=convert_pixels_to_meters(
                values=grid_cell.center, pixel_size=pixel_size
            ),
            cell_data=grid_cell.cell_data,
            fill_fraction_reference=grid_cell.fill_fraction,
            best_score=0.0,
            angle_deg=0.0,
            center_comparison=convert_pixels_to_meters(
                values=(0, 0), pixel_size=pixel_size
            ),
            is_congruent=False,  # TODO: We shouldn't set this here
            meta_data=CellMetaData(
                is_outlier=False, residual_angle_deg=0.0, position_error=(0, 0)
            ),  # TODO: We shouldn't set this here
        )
        output.append(dummy)

    return output


def fine_registration(
    comparison_mark: ProcessedMark, cells: Iterable[Cell]
) -> list[Cell]:
    """TODO: Implement function."""
    return list(cells)
