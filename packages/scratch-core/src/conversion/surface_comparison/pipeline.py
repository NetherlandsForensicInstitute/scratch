from conversion.surface_comparison.cell_registration_core import register_cells
from conversion.surface_comparison.cmc_classification import (
    classify_congruent_cells,
)
from conversion.surface_comparison.models import (
    SurfaceMap,
    ComparisonResult,
    ComparisonParams,
)


def run_comparison_pipeline(
    reference_map: SurfaceMap,
    reference_map_processed: SurfaceMap,
    comparison_map: SurfaceMap,
    comparison_map_processed: SurfaceMap,
    params: ComparisonParams,
) -> ComparisonResult:
    """
    Execute the NIST Congruent Matching Cells (CMC) pipeline.

    The pipeline runs two steps:

    1. **Per-cell registration** — the reference is divided into a symmetric
       grid; each cell undergoes a registration against the
       comparison surface

    2. **CMC classification** — median procedure 6 with ESD outlier rejection
       identifies the subset of cells whose registration parameters share a
       common consensus (``classify_congruent_cells``).

    :param reference_map: The fixed surface map.
    :param comparison_map: The moving surface map.
    :param params: CMC algorithm parameters.
    :returns: :class:`ComparisonResult` with per-cell results and CMC count.
    """
    result = ComparisonResult()

    result.cells = register_cells(
        reference_map,
        comparison_map,
        params,
        reference_map_processed,
        comparison_map_processed,
    )

    # classify_congruent_cells uses reference_map.global_center as the center
    # of rotation when computing position residuals, matching Map1.vCenterG
    # in the MATLAB implementation.
    classify_congruent_cells(result, params, reference_map.global_center)

    result.update_summary()
    return result
