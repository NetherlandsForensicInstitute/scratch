from __future__ import annotations

from conversion.surface_comparison.cell_registration.geometry import map_canvas_to_image
from conversion.surface_comparison.cell_registration.models import Match
from conversion.surface_comparison.models import Cell, CellMetaData, GridCell
from conversion.surface_comparison.utils import convert_pixels_to_meters


def record_results(
    grid_cells: list[GridCell],
    results: list[Match],
    padded_shape: tuple[int, ...],
    cell_width: int,
    cell_height: int,
) -> None:
    """
    Map each match from the padded canvas back onto its grid cell.

    :param grid_cells: Grid cells whose ``grid_search_params`` to update.
    :param results: One Match per grid cell (in the same order).
    :param padded_shape: Shape of the padded comparison canvas used for matching.
    :param cell_width: Width of one grid cell in pixels.
    :param cell_height: Height of one grid cell in pixels.
    """
    for grid_cell, match in zip(grid_cells, results, strict=True):
        center_x, center_y = map_canvas_to_image(
            match.x,
            match.y,
            (cell_height, cell_width),
            (padded_shape[0], padded_shape[1]),
            match.angle_deg,
        )
        grid_cell.grid_search_params.update(
            score=match.score,
            angle=match.angle_deg,
            center_x=center_x - cell_width,  # Undo the padding
            center_y=center_y - cell_height,
        )


def convert_grid_cell_to_cell(grid_cell: GridCell, pixel_size: float) -> Cell:
    """
    Convert a grid cell's registration result to a Cell in meters.

    :param grid_cell: Grid cell whose search results to convert.
    :param pixel_size: Pixel size in meters (assumed isotropic).
    :returns: A Cell with the grid cell's registration data expressed in meters.
    """
    return Cell(
        center_reference=convert_pixels_to_meters(
            values=grid_cell.center, pixel_size=pixel_size
        ),
        cell_size=convert_pixels_to_meters(
            values=(grid_cell.width, grid_cell.height), pixel_size=pixel_size
        ),
        fill_fraction_reference=grid_cell.fill_fraction,
        best_score=grid_cell.grid_search_params.score,
        angle_deg=grid_cell.grid_search_params.angle,
        center_comparison=convert_pixels_to_meters(
            values=(
                grid_cell.grid_search_params.center_x,
                grid_cell.grid_search_params.center_y,
            ),
            pixel_size=pixel_size,
        ),
        is_congruent=False,  # TODO: We shouldn't set this here?
        meta_data=CellMetaData(
            is_outlier=False, residual_angle_deg=0.0, position_error=(0, 0)
        ),  # TODO: We shouldn't set this here?
    )
