import numpy as np
from loguru import logger

from container_models.scan_image import ScanImage
from conversion.resample import resample_scan_image_and_mask
from conversion.surface_comparison.cell_registration.match_cells import match_cells
from conversion.surface_comparison.cmc_consensus.pipeline import (
    classify_congruent_cells_consensus,
)
from conversion.surface_comparison.grid import generate_grid
from conversion.surface_comparison.models import (
    RESAMPLE_METHOD,
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

    1. **Resample** — the comparison image is resampled to the pixel size of the reference image so both
        share a common coordinate grid.
    2. **Generate grid** — a centered rectangular grid of cells is placed over the reference image; cells with
        insufficient valid data are discarded.
    3. **Coarse-to-fine registration** — the pair is downsampled for an exhaustive coarse sweep, then each
        cell is refined locally at full resolution. See :func:`match_cells`.
    4. **CMC classification** — consensus angle and translation are estimated across all cells and each cell is
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
    reference_image = reference_mark.filtered_mark.scan_image
    comparison_image = comparison_mark.filtered_mark.scan_image

    # Step 1: Resample comparison so that both have the same pixel size
    logger.debug("starting resample")
    comparison_image, _ = resample_scan_image_and_mask(
        scan_image=comparison_image,
        target_scale=reference_image.scale_x,  # Assumes isotropic images
        # Upsampling is allowed: the search requires both images on one grid, and clipping the
        # factor at 1.0 would silently leave a coarser comparison image at its own scale.
        only_downsample=False,
        preserve_aspect_ratio=True,
        method=RESAMPLE_METHOD,
    )

    # Step 2: Generate grid cells
    logger.debug("starting grid generation")
    grid_cells = generate_grid(
        scan_image=reference_image,
        cell_size=reference_mark.filtered_mark.mark_type.cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
        nan_fill_value=resolve_nan_fill_value(reference_image, params),
    )

    # Step 3: Coarse-to-fine registration
    logger.debug("starting cell registration")
    cells = match_cells(
        grid_cells=grid_cells,
        reference_image=reference_image,
        comparison_image=comparison_image,
        params=params,
    )

    # Step 4: CMC classification
    logger.debug("starting cmc classification")
    return classify_congruent_cells_consensus(
        cells=cells, params=params, reference_center=reference_image.center_meters
    )


def resolve_nan_fill_value(
    reference_image: ScanImage, params: ComparisonParams
) -> float | None:
    """
    Turn ``template_nan_fill_strategy`` into the concrete value every template will be filled with.

    ``None`` means "each cell's own valid-pixel mean"; see
    :func:`~conversion.surface_comparison.template_fill.fill_template_nan`.
    """
    if params.template_nan_fill_strategy != "global_mean":
        return None
    nan_fill_value = float(np.nanmean(reference_image.data))
    logger.debug(
        "Using global mean ({:.4f}) for NaN filling (template_nan_fill_strategy=global_mean)",
        nan_fill_value,
    )
    return nan_fill_value
