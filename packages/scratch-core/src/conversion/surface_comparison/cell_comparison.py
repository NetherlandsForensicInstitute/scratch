import numpy as np
from skimage.feature import match_template
from skimage.transform import rotate

from container_models.base import FloatArray1D, FloatArray2D
from conversion.surface_comparison.models import (
    SurfaceMap,
    CellResult,
    ComparisonParams,
)
from conversion.surface_comparison.cell_grid import (
    find_optimal_cell_origin,
    generate_cell_centers,
)


def run_cell_comparison(
    reference_map: SurfaceMap,
    comparison_map: SurfaceMap,
    params: ComparisonParams,
    initial_rotation: float = 0.0,
) -> list[CellResult]:
    """
    Execute the per-cell correlation search localized around an initial rotation.

    :param reference_map: The fixed surface map.
    :param comparison_map: The moving surface map.
    :param params: Algorithm parameters containing search ranges and thresholds.
    :param initial_rotation: The global rotation consensus in radians from coarse alignment.
    :returns: A list of CellResult objects for all valid cells found.
    """
    origin = find_optimal_cell_origin(reference_map, params)
    centers = generate_cell_centers(reference_map, origin, params)

    results = []
    for center in centers:
        result = _process_single_cell(
            center, reference_map, comparison_map, params, initial_rotation
        )
        if result is not None:
            results.append(result)
    return results


def _process_single_cell(
    center: FloatArray1D,
    reference_map: SurfaceMap,
    comparison_map: SurfaceMap,
    params: ComparisonParams,
    initial_rotation: float,
) -> CellResult | None:
    """
    Extracts a patch and attempts local registration.

    :param center: [x, y] center of the cell in micrometers.
    :param reference_map: The reference SurfaceMap.
    :param comparison_map: The comparison SurfaceMap.
    :param params: Comparison parameters.
    :param initial_rotation: Global rotation consensus in radians.
    :returns: CellResult if cell contains sufficient data, else None.
    """
    pixel_spacing = reference_map.pixel_spacing
    cell_pixel_size = (params.cell_size / pixel_spacing).astype(int)

    # Extract reference patch
    pixel_y = int(round(center[1] / pixel_spacing[1] - cell_pixel_size[1] / 2))
    pixel_x = int(round(center[0] / pixel_spacing[0] - cell_pixel_size[0] / 2))
    rows, cols = reference_map.height_map.shape
    if (
        pixel_y < 0
        or pixel_x < 0
        or pixel_y + cell_pixel_size[1] > rows
        or pixel_x + cell_pixel_size[0] > cols
    ):
        return None

    patch = reference_map.height_map[
        pixel_y : pixel_y + cell_pixel_size[1], pixel_x : pixel_x + cell_pixel_size[0]
    ]
    # Check fill fraction
    valid_mask = ~np.isnan(patch)
    fill_fraction = np.count_nonzero(valid_mask) / patch.size
    if fill_fraction < params.minimum_fill_fraction:
        return None

    return _find_optimal_registration(
        center, patch, comparison_map, params, fill_fraction, initial_rotation
    )


def _find_optimal_registration(
    center: FloatArray1D,
    patch: FloatArray2D,
    comparison_map: SurfaceMap,
    params: ComparisonParams,
    fill_fraction: float,
    initial_rotation: float,
) -> CellResult:
    """
    Finds the best local match for a reference cell within the comparison surface.

    :param center: Center of cell in reference (micrometers).
    :param patch: 2D array of reference height data.
    :param comparison_map: The comparison SurfaceMap.
    :param params: Comparison parameters.
    :param fill_fraction: Fraction of non-NaN pixels in the patch.
    :param initial_rotation: Global rotation consensus in radians.
    :returns: Optimal CellResult for this local area.
    """
    # Convert initial rotation to degrees for the search grid
    initial_deg = np.degrees(initial_rotation)

    # Define search range relative to the global consensus
    angles = np.arange(
        initial_deg + params.search_angle_min,
        initial_deg + params.search_angle_max + params.search_angle_step,
        params.search_angle_step,
    )

    # Pre-clean reference patch (zero-mean normalization)
    clean_patch = np.nan_to_num(patch, nan=float(np.nanmean(patch)))
    clean_patch -= np.mean(clean_patch)
    comp_data_full = np.nan_to_num(
        comparison_map.height_map,
        nan=float(np.nanmean(comparison_map.height_map)),
    )

    ph, pw = clean_patch.shape
    best_score = -1.0
    best_angle_rad = 0.0
    best_center_comp = np.zeros(2)
    for angle_deg in angles:
        # Rotate comparison image
        rotated_comp = (
            rotate(comp_data_full, float(-angle_deg), preserve_range=True)
            if not np.isclose(angle_deg, 0.0)
            else comp_data_full
        )

        # Cross-correlation map
        accf_map = match_template(rotated_comp, clean_patch, pad_input=False)
        idx_y, idx_x = np.unravel_index(np.argmax(accf_map), accf_map.shape)
        score = float(accf_map[idx_y, idx_x])

        if score > best_score:
            best_score = score
            best_angle_rad = np.radians(angle_deg)
            # Match index refers to top-left; calculate center in microns
            match_center_px = np.array([idx_x + (pw - 1) / 2.0, idx_y + (ph - 1) / 2.0])
            best_center_comp = match_center_px * comparison_map.pixel_spacing

    return CellResult(
        center_reference=center,
        center_comparison=best_center_comp,
        registration_angle=best_angle_rad,
        area_cross_correlation_function_score=best_score,
        reference_fill_fraction=fill_fraction,
    )
