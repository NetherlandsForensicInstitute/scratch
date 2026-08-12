import numpy as np
from loguru import logger

from container_models.scan_image import ScanImage
from conversion.resample import (
    SCALE_MATCH_RTOL,
    Interpolation,
    get_scaling_factors,
    resample_nan_aware,
)
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

#: Interpolation for shrinking an image: it averages every source pixel rather than sampling a
#: subset, which is what keeps aliasing out of a downsampled surface. Also what every other resample
#: in this pipeline gets by default, since they only ever shrink.
DOWNSAMPLE_INTERPOLATION: Interpolation = "area"
#: Interpolation for growing an image. Not ``area``, which cv2 degenerates to nearest-neighbor when
#: zooming in, and not ``cubic``, whose outer taps carry negative weights: those would let
#: :func:`~conversion.resample.resize_nan_aware` divide by a near-zero or negative coverage at the
#: edge of a missing-data hole, exactly where the data is already weakest.
UPSAMPLE_INTERPOLATION: Interpolation = "linear"


def select_interpolation(factors: tuple[float, float]) -> Interpolation:
    """
    Pick the interpolation to resize by, from the direction the image is being resized in.

    A factor above 1.0 shrinks that axis. Shrinking wants :data:`DOWNSAMPLE_INTERPOLATION` so no
    source pixel is skipped; as soon as one axis grows, the whole resize has to use
    :data:`UPSAMPLE_INTERPOLATION`, since cv2 takes a single flag for both axes.

    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: The interpolation name to resample with.
    """
    is_shrinking = all(factor >= 1.0 for factor in factors)
    return DOWNSAMPLE_INTERPOLATION if is_shrinking else UPSAMPLE_INTERPOLATION


def resample_to_scale(image: ScanImage, target_scale: float) -> ScanImage:
    """
    Put *image* on a pixel grid of *target_scale*, NaN-aware and in either direction.

    Deliberately not :func:`~conversion.resample.resample_scan_image_and_mask`: that one clips the
    factor at 1.0 by default, which would silently leave a coarser comparison image at its own
    scale, and it resizes the way every other pipeline wants rather than the way this one needs.
    Growing an image is allowed here precisely because the search requires both marks on one grid,
    and that is what makes the interpolation depend on the direction.

    :param image: The image to put on the target grid.
    :param target_scale: Target scale (= pixel size in meters), assumed isotropic.
    :returns: The resampled image, or *image* itself when it is already on that grid.
    """
    factors = get_scaling_factors(
        scales=(image.scale_x, image.scale_y), target_scale=target_scale
    )
    if np.allclose(factors, 1.0, rtol=SCALE_MATCH_RTOL, atol=0.0):
        return image
    return ScanImage(
        data=resample_nan_aware(image.data, factors, select_interpolation(factors)),
        scale_x=image.scale_x * factors[0],
        scale_y=image.scale_y * factors[1],
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
    comparison_image = resample_to_scale(comparison_image, reference_image.scale_x)

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
