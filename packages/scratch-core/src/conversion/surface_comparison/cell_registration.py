from collections.abc import Iterable

from container_models.scan_image import ScanImage
from conversion.surface_comparison.grid import GridCell, extract_patch, GridSearchParams
from conversion.surface_comparison.models import ComparisonParams, Cell, CellMetaData
import numpy as np

from scipy.ndimage import rotate

from conversion.surface_comparison.pipeline import ProcessedMark


def _rotate_scan_image(scan_image: ScanImage, angle: float) -> ScanImage:
    return scan_image.model_copy(
        update={"data": rotate(scan_image.data, angle=angle, cval=np.nan)}
    )


def _find_best_translation(
    scan_image: ScanImage,
    cell: GridCell,
    min_fill_fraction: float,
    nan_fill_value: float,
) -> tuple[float, int, int]:
    # TODO: Implement correct implementation
    if np.isnan(cell.cell_data).any():
        raise ValueError("Cell data cannot contain NaNs.")

    max_score = float("-inf")
    best_x, best_y = 0, 0
    for y in range(scan_image.height - cell.height // 2):
        for x in range(scan_image.width - cell.width // 2):
            patch = extract_patch(
                scan_image=scan_image, coordinates=(x, y), size=cell.size
            )
            fill_fraction = 1 - np.isnan(patch).sum() / patch.size
            if fill_fraction < min_fill_fraction:
                # If not enough data, go to next point
                continue
            patch = patch.copy()
            # Replace NaNs with fill value so that we can compute a score
            patch[np.isnan(patch)] = nan_fill_value
            score = np.corrcoef(cell.cell_data, patch)
            if score > max_score:
                best_x, best_y = x, y
                max_score = score
    return max_score, best_x, best_y


def _pixels_to_meters(
    coordinates: tuple[int, int], pixel_size: tuple[float, float]
) -> tuple[float, float]:
    # TODO: remove this function if possible?
    return coordinates[0] * pixel_size[0], coordinates[1] * pixel_size[1]


def coarse_registration(
    grid_cells: Iterable[GridCell],
    comparison_image: ScanImage,
    params: ComparisonParams,
    fill_value_reference: float,
) -> list[Cell]:
    # TODO: Write docstring
    angle = params.search_angle_min
    fill_value_comparison = float(np.nanmean(comparison_image))
    pixel_size = comparison_image.scale_x, comparison_image.scale_y

    while angle < params.search_angle_max:
        # Rotate the comparison image by `-angle` degrees.
        # This is equivalent to rotating the reference patch by `angle` degrees.
        rotated = _rotate_scan_image(scan_image=comparison_image, angle=-angle)

        for grid_cell in grid_cells:
            # TODO: Rewrite and optimize this part
            # Deep copy cell and fill NaN values with numerical value
            # We cannot use `grid_cell` directly yet for this since we neet its fill fraction later
            cell_copy = GridCell(
                center=grid_cell.center,
                size=grid_cell.size,
                cell_data=grid_cell.cell_data.copy(),
                grid_search_params=GridSearchParams(),
            )
            cell_copy.fill_nans(fill_value=fill_value_reference)

            score, x, y = _find_best_translation(
                scan_image=rotated,
                cell=cell_copy,
                min_fill_fraction=params.minimum_fill_fraction,  # TODO: use different fill fraction for comparison
                nan_fill_value=fill_value_comparison,
            )
            if score > grid_cell.grid_search_params.score:
                # Update search parameters
                grid_cell.grid_search_params.update(score=score, angle=angle, x=x, y=y)

        angle += params.search_angle_step

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
