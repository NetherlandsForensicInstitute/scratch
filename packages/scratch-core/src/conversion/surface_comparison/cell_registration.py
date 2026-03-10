from collections.abc import Iterable

from container_models.scan_image import ScanImage
from conversion.surface_comparison.grid import GridCell
from conversion.surface_comparison.models import ComparisonParams, Cell, CellMetaData
import numpy as np

from scipy.ndimage import rotate

from conversion.surface_comparison.pipeline import ProcessedMark


def _rotate_scan_image(scan_image: ScanImage, angle: float) -> ScanImage:
    return scan_image.model_copy(
        update={"data": rotate(scan_image.data, angle=angle, cval=np.nan)}
    )


def _pixels_to_meters(
    coordinates: tuple[int, int], pixel_size: tuple[float, float]
) -> tuple[float, float]:
    # TODO: Remove this function if possible?
    return coordinates[0] * pixel_size[0], coordinates[1] * pixel_size[1]


def coarse_registration(
    grid_cells: Iterable[GridCell],
    comparison_image: ScanImage,
    params: ComparisonParams,
    fill_value_reference: float,  # Fill value for NaNs in the grid cell data
) -> list[Cell]:
    # TODO: Implement this
    pixel_size = comparison_image.scale_x, comparison_image.scale_y
    output = []
    for grid_cell in grid_cells:
        cell = Cell(
            center_reference=_pixels_to_meters(
                coordinates=grid_cell.center, pixel_size=pixel_size
            ),
            cell_data=grid_cell.cell_data,
            fill_fraction_reference=grid_cell.fill_fraction,
            best_score=grid_cell.grid_search_params.score,
            angle_deg=grid_cell.grid_search_params.angle,
            center_comparison=_pixels_to_meters(
                coordinates=(
                    grid_cell.grid_search_params.x,
                    grid_cell.grid_search_params.y,
                ),
                pixel_size=pixel_size,
            ),
            is_congruent=False,  # TODO: We shouldn't set this here
            meta_data=CellMetaData(
                is_outlier=False, residual_angle_deg=0.0, position_error=(0, 0)
            ),  # TODO: We shouldn't set this here
        )
        output.append(cell)
    return output


def fine_registration(
    comparison_mark: ProcessedMark, cells: Iterable[Cell]
) -> list[Cell]:
    # TODO: Implement this
    return list(cells)
