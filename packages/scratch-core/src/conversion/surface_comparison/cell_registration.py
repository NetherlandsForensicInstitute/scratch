from collections.abc import Iterable
import cv2
from container_models.base import FloatArray2D, BinaryMask
from container_models.scan_image import ScanImage
from conversion.surface_comparison.grid import GridCell
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
    CellMetaData,
    ProcessedMark,
)
import numpy as np
from skimage.transform import rotate
from conversion.surface_comparison.utils import (
    convert_pixels_to_meters,
)


def _get_fill_fraction_map(
    valid_pixel_mask: BinaryMask,
    cell_height: int,
    cell_width: int,
) -> FloatArray2D:
    """
    Compute a 2D map where each entry [r, c] is the fill fraction of a
    cell-sized window with its **top-left corner** at pixel (c, r), matching
    the indexing convention of ``cv2.matchTemplate``.

    Uses ``cv2.filter2D`` with ``anchor=(0, 0)`` so the kernel origin sits at
    the top-left of the window, producing O(N) box-filter computation.

    :param valid_pixel_mask: Boolean array (H, W); True where image data is valid.
    :param cell_height: Height of the cell window in pixels.
    :param cell_width: Width of the cell window in pixels.
    :returns: Float32 array (H, W) with fill fractions in [0, 1], top-left indexed.
              Entries near the bottom-right boundary are underestimates and will
              be rejected by the fill-fraction gate.
    """
    kernel = np.ones((cell_height, cell_width), dtype=np.float32) / (
        cell_height * cell_width
    )
    return cv2.filter2D(
        valid_pixel_mask.astype(np.float32),
        ddepth=-1,
        kernel=kernel,
        anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT,
    )


def _get_score_map(
    comparison_array_filled: FloatArray2D, cell: GridCell
) -> FloatArray2D:
    from skimage.feature import match_template
    score_map = match_template(image=comparison_array_filled, template=cell.cell_data, pad_input=False)
    # score_map = cv2.matchTemplate(
    #     image=comparison_array_filled.astype(np.float32),
    #     templ=cell.cell_data.astype(np.float32),
    #     method=cv2.TM_CCOEFF_NORMED,
    # )
    return score_map


def _convert_grid_cell_to_cell(grid_cell: GridCell, pixel_size: float) -> Cell:
    cell = Cell(
        center_reference=convert_pixels_to_meters(
            values=grid_cell.center, pixel_size=pixel_size
        ),
        cell_data=grid_cell.cell_data,
        fill_fraction_reference=grid_cell.fill_fraction,
        best_score=grid_cell.grid_search_params.score,
        angle_deg=grid_cell.grid_search_params.angle,
        center_comparison=convert_pixels_to_meters(
            # TODO: convert top_left to center
            values=(
                grid_cell.grid_search_params.top_left_x,
                grid_cell.grid_search_params.top_left_y,
            ),
            pixel_size=pixel_size,
        ),
        is_congruent=False,  # TODO: We shouldn't set this here
        meta_data=CellMetaData(
            is_outlier=False, residual_angle_deg=0.0, position_error=(0, 0)
        ),  # TODO: We shouldn't set this here
    )
    return cell


def coarse_registration(
    grid_cells: Iterable[GridCell],
    comparison_image: ScanImage,
    params: ComparisonParams,
    fill_value_reference: float,  # Fill value for NaNs in the grid cell data
) -> list[Cell]:
    # TODO: Implement this
    angle = params.search_angle_min
    fill_value_comparison = float(np.nanmean(comparison_image.data))
    pixel_size = comparison_image.scale_x  # Assumes isotropic image
    first_cell = list(grid_cells)[0]

    templates = []
    for grid_cell in grid_cells:
        # Deep copy cell and fill NaN values with numerical value
        cell_copy = grid_cell.copy()
        cell_copy.fill_nans(fill_value=fill_value_reference)
        templates.append(cell_copy)

    while angle < params.search_angle_max:
        # Rotate the comparison image by `-angle` degrees.
        # This is equivalent to rotating the reference patch by `angle` degrees.
        rotated = rotate(
            image=comparison_image.data,
            angle=-angle,
            cval=np.nan,
            order=0,
            resize=False,
        )
        valid_mask = ~np.isnan(rotated)
        fill_fraction_map = _get_fill_fraction_map(
            valid_pixel_mask=valid_mask,
            cell_width=first_cell.width,
            cell_height=first_cell.height,
        )
        # TODO: use different fill fraction for comparison?
        fill_fraction_mask = fill_fraction_map >= params.minimum_fill_fraction
        rotated[~valid_mask] = fill_value_comparison
        # rotated[~valid_mask] = np.random.normal(fill_value_comparison, 1e-6, size=(~valid_mask).sum())
        for grid_cell, template in zip(grid_cells, templates):
            score_map = _get_score_map(
                comparison_array_filled=rotated,
                cell=template,
            )
            sliced_mask = fill_fraction_mask[: score_map.shape[0], : score_map.shape[1]]
            # Replace non-valid values with -inf
            masked_scores = np.where(sliced_mask, score_map, -np.inf)
            best_flat_index = int(np.argmax(masked_scores))
            score = float(masked_scores.flat[best_flat_index])
            if angle == 0.0:
                import matplotlib.pyplot as plt

                # Maak een figuur met 1 rij en 2 kolommen
                fig, ax = plt.subplots(2, 2, figsize=(10, 5))

                # Toon de eerste afbeelding in de linker subplot
                ax[0][0].imshow(rotated)
                ax[0][0].set_title("rotated")
                ax[0][0].axis("off")  # Verberg de assen

                # Toon de tweede afbeelding in de rechter subplot
                ax[0][1].imshow(template.cell_data)
                ax[0][1].set_title("template")
                ax[0][1].axis("off")  # Verberg de assen

                # Toon de eerste afbeelding in de linker subplot
                ax[1][0].imshow(score_map)
                ax[1][0].set_title("score_map")
                ax[1][0].axis("off")  # Verberg de assen

                # Toon de tweede afbeelding in de rechter subplot
                ax[1][1].imshow(valid_mask)
                ax[1][1].set_title("valid_mask")
                ax[1][1].axis("off")  # Verberg de assen

                plt.tight_layout()  # Optimaliseer de tussenruimte
                plt.show()
                pass
            if score > grid_cell.grid_search_params.score:
                y, x = np.unravel_index(best_flat_index, masked_scores.shape)
                grid_cell.grid_search_params.update(
                    score=score, angle=angle, top_left_x=int(x), top_left_y=int(y)
                )
        angle += params.search_angle_step

    output = [
        _convert_grid_cell_to_cell(grid_cell=c, pixel_size=pixel_size)
        for c in grid_cells
    ]
    return output


def fine_registration(
    comparison_mark: ProcessedMark, cells: Iterable[Cell]
) -> list[Cell]:
    """TODO: Implement function."""
    return list(cells)
