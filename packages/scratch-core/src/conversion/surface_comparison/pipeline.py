import numpy as np

from conversion.resample import resample_scan_image_and_mask
from conversion.surface_comparison.cell_registration import (
    coarse_registration,
    fine_registration,
)
from conversion.surface_comparison.cmc_classification import classify_congruent_cells
from conversion.surface_comparison.grid import generate_grid
from conversion.surface_comparison.models import (
    ComparisonParams,
    ComparisonResult,
    ProcessedMark,
)


def compare_surfaces(
    refence_mark: ProcessedMark,
    comparison_mark: ProcessedMark,
    params: ComparisonParams,
) -> ComparisonResult:
    """TODO: Write docstring."""

    # Get the filtered images for the CMC pipeline
    reference_image = refence_mark.filtered_mark.scan_image
    comparison_image = comparison_mark.filtered_mark.scan_image

    # Step 1: Resample comparison so that both have the same pixel size
    pixel_size = reference_image.scale_x  # Assumes isotropic image
    comparison_image, _ = resample_scan_image_and_mask(
        scan_image=comparison_image, target_scale=pixel_size, preserve_aspect_ratio=True
    )

    # Step 2: Generate grid cells
    grid_cells = generate_grid(scan_image=reference_image, params=params)

    # Step 3: Coarse registration
    fill_value_reference = float(np.nanmean(reference_image.data))
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
        fill_value_reference=fill_value_reference,
    )

    # Step 4: Fine registration
    cells = fine_registration(comparison_mark=comparison_mark, cells=cells)

    # Step 5: CMC classification
    comparison_result = classify_congruent_cells(
        cells=cells, params=params, reference_center=reference_image.center_meters
    )
    return comparison_result
