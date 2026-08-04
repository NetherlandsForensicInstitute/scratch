from loguru import logger

from conversion.surface_comparison.cell_registration.match_cells import match_cells
from conversion.surface_comparison.cmc_consensus.pipeline import (
    classify_congruent_cells_consensus,
)
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

    Executes the four-step pipeline:

    1. **Generate grid** — a centered rectangular grid of cells is placed over the reference image; cells with
        insufficient valid data are discarded.
    2. **Coarse-to-fine registration** — the reference and comparison images (which may be at different native
        pixel scales) are brought to a shared scale and downsampled for an exhaustive coarse sweep, then each
        cell is refined locally at full resolution. See :func:`match_cells` for the full description.
    3. **CMC classification** — consensus angle and translation are estimated across all cells and each cell is
        labeled as congruent or not.

    Both marks are expected to have already been pre-processed (leveled and band-pass filtered);
    only the ``filtered_mark`` image is currently used by the pipeline.

    :param reference_mark: Pre-processed reference mark; its filtered scan image defines the grid and coordinate system.
    :param comparison_mark: Pre-processed comparison mark to register against the reference.
    :param params: Algorithm parameters controlling cell size, fill-fraction thresholds, angle sweep, coarse/fine
        search configuration, and CMC classification thresholds.
    :returns: A :class:`ComparisonResult` containing per-cell registration results, the consensus rotation and
        translation, and CMC counts.
    """

    # Get the filtered images for the CMC pipeline
    reference_image = reference_mark.filtered_mark.scan_image
    comparison_image = comparison_mark.filtered_mark.scan_image

    # Step 1: Generate grid cells
    logger.debug("starting grid generation")
    grid_cells = generate_grid(
        scan_image=reference_image,
        cell_size=params.cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
    )

    # Step 2: Coarse-to-fine registration (scale alignment happens inside match_cells)
    logger.debug("starting cell registration")
    cells = match_cells(
        grid_cells=grid_cells,
        reference_image=reference_image,
        comparison_image=comparison_image,
        params=params,
    )

    # Step 3: CMC classification
    logger.debug("starting cmc classification")
    comparison_result = classify_congruent_cells_consensus(
        cells=cells, params=params, reference_center=reference_image.center_meters
    )
    return comparison_result
