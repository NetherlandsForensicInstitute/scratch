from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import (
    ComparisonResult,
    ComparisonParams,
)
from conversion.surface_comparison.cell_registration import register_cells
from conversion.surface_comparison.cmc_classification import classify_congruent_cells


def run_comparison_pipeline(
    reference_map: ScanImage, comparison_map: ScanImage, params: ComparisonParams
) -> ComparisonResult:
    """
    Execute the NIST Congruent Matching Cells (CMC) pipeline.

    The pipeline runs two steps:

    1. **Per-cell registration** — the reference is divided into a symmetric
       grid; each cell undergoes a two-stage registration against the comparison
       surface via the MATLAB-faithful ``cell_corr_analysis`` engine:

       - Stage 1 (coarse angular sweep): evaluates ACCF at discrete angles.
       - Stage 2 (ECC gradient): refines [dx, dy, θ] iteratively.

    2. **CMC classification** — median procedure 6 with ESD outlier rejection
       identifies the subset of cells whose registration parameters share a
       common consensus (``classify_congruent_cells``).

    :param reference_map: The fixed surface map.
    :param comparison_map: The moving surface map.
    :param params: CMC algorithm parameters.
    :returns: :class:`ComparisonResult` with per-cell results and CMC count.
    """
    cells = register_cells(reference_map, comparison_map, params)

    # classify_congruent_cells uses reference_map.global_center as the center
    # of rotation when computing position residuals, matching Map1.vCenterG
    # in the MATLAB implementation.
    global_center = reference_map.global_center
    rotation_center: tuple[float, float] = (
        float(global_center[0]),
        float(global_center[1]),
    )

    return classify_congruent_cells(cells, params, rotation_center)
