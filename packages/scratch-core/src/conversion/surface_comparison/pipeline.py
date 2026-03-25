from conversion.resample import resample_scan_image_and_mask


from conversion.surface_comparison.cell_registration.core import (
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
    reference_mark: ProcessedMark,
    comparison_mark: ProcessedMark,
    params: ComparisonParams,
) -> ComparisonResult:
    """
    Run the full CMC pipeline to compare two cartridge-case surface marks.

    Executes the five-step pipeline:

    1. **Resample** — the comparison image is resampled to match the pixel size of the reference image so both
        share a common coordinate grid.
    2. **Generate grid** — a centered rectangular grid of cells is placed over the reference image; cells with
        insufficient valid data are discarded.
    3. **Coarse registration** — each reference cell is matched against the comparison image over a configurable
        angle sweep to find the best-scoring translation and rotation.
    4. **Fine registration** — currently a pass-through stub; intended for sub-pixel refinement.
    5. **CMC classification** — consensus angle and translation are estimated across all cells and each cell is
        labeled as congruent or not.

    Both marks are expected to have already been pre-processed (leveled and band-pass filtered);
    only the ``filtered_mark`` image is currently used by the pipeline.

    :param reference_mark: Pre-processed reference mark; its filtered scan image defines the grid and coordinate system.
    :param comparison_mark: Pre-processed comparison mark to register against the reference.
    :param params: Algorithm parameters controlling cell size, fill-fraction thresholds, angle sweep, and CMC
        classification thresholds.
    :returns: A :class:`ComparisonResult` containing per-cell registration results, the consensus rotation and
        translation, and CMC counts.
    """

    # Get the filtered images for the CMC pipeline
    reference_image = reference_mark.filtered_mark.scan_image
    comparison_image = comparison_mark.filtered_mark.scan_image

    # Step 1: Resample comparison so that both have the same pixel size
    pixel_size = reference_image.scale_x  # Assumes isotropic image
    comparison_image, _ = resample_scan_image_and_mask(
        scan_image=comparison_image, target_scale=pixel_size, preserve_aspect_ratio=True
    )

    # Step 2: Generate grid cells
    grid_cells = generate_grid(
        scan_image=reference_image,
        cell_size=params.cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
    )

    # Step 3: Coarse registration
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
    )

    # Step 4: Fine registration
    cells = fine_registration(comparison_mark=comparison_mark, cells=cells)

    # Step 5: CMC classification
    comparison_result = classify_congruent_cells(
        cells=cells, params=params, reference_center=reference_image.center_meters
    )
    return comparison_result
