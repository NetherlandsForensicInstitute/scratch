from conversion.surface_comparison.models import (
    SurfaceMap,
    ComparisonResult,
    ComparisonParams,
)
from conversion.surface_comparison.cell_registration import register_cells
from conversion.surface_comparison.congruent_cell_classifier import (
    classify_congruent_cells,
)


def run_comparison_pipeline(
    reference_map: SurfaceMap, comparison_map: SurfaceMap, params: ComparisonParams
) -> ComparisonResult:
    """
    Execute the NIST Congruent Matching Cells (CMC) pipeline.

    The pipeline runs two steps:

    1. **Per-cell registration** — the reference is divided into a symmetric
       grid; each cell undergoes a three-stage registration against the
       comparison surface:

       - Stage 1 (coarse angular sweep): the full comparison image is rotated
         once per candidate angle and every cell is matched via normalised
         cross-correlation (``match_template``), mirroring
         ``cell_corr_angle``.
       - Stage 2 (sub-pixel FFT CC): ``phase_cross_correlation`` pins the
         translation to sub-pixel precision at the best angle from Stage 1,
         mirroring ``maps_register_corr``.
       - Stage 3 (ECC gradient): the Enhanced Correlation Coefficient
         algorithm (Evangelidis & Psarakis 2008) refines ``[dx, dy, θ]``
         iteratively, mirroring ``maps_register_fine`` with
         ``regAlgorithmFine='gradient'``.

    2. **CMC classification** — median procedure 6 with ESD outlier rejection
       identifies the subset of cells whose registration parameters share a
       common consensus (``classify_congruent_cells``).

    :param reference_map: The fixed surface map.
    :param comparison_map: The moving surface map.
    :param params: CMC algorithm parameters.
    :returns: :class:`ComparisonResult` with per-cell results and CMC count.
    """
    result = ComparisonResult()

    result.cells = register_cells(reference_map, comparison_map, params)

    # classify_congruent_cells uses reference_map.global_center as the center
    # of rotation when computing position residuals, matching Map1.vCenterG
    # in the MATLAB implementation.
    classify_congruent_cells(result, params, reference_map.global_center)

    result.update_summary()
    return result
