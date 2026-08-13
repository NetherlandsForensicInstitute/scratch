import numpy as np
from loguru import logger

from container_models.scan_image import ScanImage
from conversion.resample import resample_scan_image_to_scale
from conversion.surface_comparison.cell_registration.results import (
    convert_grid_cell_to_cell,
    record_results,
)
from conversion.surface_comparison.cell_registration.search import (
    find_best_matches,
    get_uniform_cell_shape,
    search_candidates,
)
from conversion.surface_comparison.cell_registration.stage_builders import (
    build_angle_sweep,
    build_coarse_stage,
    build_full_resolution_stage,
    compute_cap_factor,
)
from conversion.surface_comparison.cell_registration.stages import run_fine_stage
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

    Executes the pipeline:

    1. **Resample** — the comparison image is resampled to the pixel size of the reference image.
    2. **Generate grid** — a centered rectangular grid of cells is placed over the reference image.
    3. **Build the full-resolution stage** — the scale-aligned comparison image and reference templates, padded.
    4. **Coarse sweep** — if images are larger than ``params.coarse_target_size``, an exhaustive translation + rotation
        search is performed on downsampled images.
    5. **Fine refinement** — local search around each coarse candidate at full resolution.
    6. **Gather results** — matches are mapped back onto grid cells and converted to Cell instances.
    7. **CMC classification** — consensus angle and translation are estimated across all cells and each cell
        is labeled as congruent or not.

    Both marks are expected to have already been pre-processed (leveled and band-pass filtered);
    only the ``filtered_mark`` image is currently used by the pipeline.

    :param reference_mark: Pre-processed reference mark; its filtered scan image defines the grid and coordinate system.
    :param comparison_mark: Pre-processed comparison mark to register against the reference.
    :param params: Algorithm parameters controlling cell size, fill-fraction thresholds, angle sweep, coarse/fine
        search configuration, and CMC classification thresholds.
    :returns: A ComparisonResult containing per-cell registration results, the consensus rotation and
        translation, and CMC counts.
    """
    reference_image = reference_mark.filtered_mark.scan_image
    comparison_image_original = comparison_mark.filtered_mark.scan_image

    # Step 1: Resample comparison to reference scale (for the fine stage)
    logger.debug("starting resample")
    comparison_image_full = resample_scan_image_to_scale(
        comparison_image_original, reference_image.scale_x
    )

    # Step 2: Generate grid cells
    logger.debug("starting grid generation")
    grid_cells = generate_grid(
        scan_image=reference_image,
        cell_size=reference_mark.filtered_mark.mark_type.cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
        nan_fill_value=resolve_nan_fill_value(reference_image, params),
    )
    if not grid_cells:
        # classify_congruent_cells_consensus is documented to reject an empty cells list, so
        # build the trivial empty result directly rather than call it out of contract.
        logger.debug("no grid cells generated. skipping cell registration.")
        return ComparisonResult(
            cells=[], estimated_rotation=0.0, estimated_translation=(0.0, 0.0)
        )

    # Step 3: Build the full-resolution stage
    logger.debug("starting cell registration")
    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    full_stage = build_full_resolution_stage(
        comparison_image_full, grid_cells, cell_width, cell_height
    )
    cap_factor = compute_cap_factor(
        reference_image,
        comparison_image_full,
        cell_width,
        cell_height,
        params.coarse_target_size,
    )
    angles = build_angle_sweep(params)

    if cap_factor > 1.0:
        # Step 4: Coarse sweep — single pass from ORIGINAL images (no interpolation chaining)
        coarse_stage = build_coarse_stage(
            comparison_image_original,
            reference_image,
            grid_cells,
            cap_factor,
        )
        candidates = search_candidates(
            coarse_stage.image,
            coarse_stage.templates,
            angles,
            params.minimum_fill_fraction,
            coarse_stage.fill_value,
            n_candidates=params.n_candidates,
            template_batch_size=params.template_batch_size,
            angle_batch_size=params.angle_batch_size,
        )

        # Step 5: Fine refinement
        matches = run_fine_stage(
            image_full=full_stage.image,
            templates_full=full_stage.templates,
            candidates=candidates,
            coarse_cell_shape=get_uniform_cell_shape(coarse_stage.templates),
            coarse_image_shape=coarse_stage.image.shape,
            cap_factor=cap_factor,
            angles=angles,
            position_margin=params.fine_n_pixels,
            angle_margin_degrees=params.fine_m_degrees,
            minimum_fill_fraction=params.minimum_fill_fraction,
            fill_value_full=full_stage.fill_value,
            fine_batch_size=params.fine_batch_size,
        )
    else:
        # Steps 4-5: images already fit the coarse target, so coarse and fine would search the
        # same resolution — skip the coarse stage and run one exhaustive pass instead of the
        # same work twice.
        logger.debug(
            "cap_factor <= 1.0: skipping the coarse stage for a single exhaustive pass"
        )
        matches = find_best_matches(
            full_stage.image,
            full_stage.templates,
            angles,
            params.minimum_fill_fraction,
            full_stage.fill_value,
            template_batch_size=params.template_batch_size,
            angle_batch_size=params.angle_batch_size,
        )

    # Step 6: Gather results and convert to Cell instances
    record_results(grid_cells, matches, full_stage.image.shape, cell_width, cell_height)
    cells = [
        convert_grid_cell_to_cell(
            grid_cell=grid_cell, pixel_size=reference_image.scale_x
        )
        for grid_cell in grid_cells
    ]

    # Step 7: CMC classification
    logger.debug("starting cmc classification")
    return classify_congruent_cells_consensus(
        cells=cells, params=params, reference_center=reference_image.center_meters
    )


def resolve_nan_fill_value(
    reference_image: ScanImage, params: ComparisonParams
) -> float | None:
    """
    Turn ``template_nan_fill_strategy`` into the concrete value every template will be filled with.

    ``None`` means "each cell's own valid-pixel mean"; see conversion.surface_comparison.template_fill.fill_template_nan.

    :param reference_image: Reference scan image; its global mean is used for ``global_mean`` strategy.
    :param params: Comparison parameters specifying the fill strategy.
    :returns: A fill value for ``global_mean``, or ``None`` for ``local_mean``.
    """
    # TODO: ``local_mean`` needs the masked NCC of Padfield (2012) to be correct. The score denominator in
    # conversion.surface_comparison.cell_registration.scoring.build_correlation_basis normalizes over the
    # whole window, while the numerator only covers the overlap of the two validity masks, so scores are
    # deflated in proportion to how empty a cell is. ``global_mean`` wins today because it happens
    # to offset that.
    if params.template_nan_fill_strategy != "global_mean":
        return None
    nan_fill_value = float(np.nanmean(reference_image.data))
    logger.debug(
        "Using global mean ({:.4f}) for NaN filling (template_nan_fill_strategy=global_mean)",
        nan_fill_value,
    )
    return nan_fill_value
