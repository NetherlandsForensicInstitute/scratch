from conversion.surface_comparison.models import (
    SurfaceMap,
    ComparisonResult,
    ComparisonParams,
)
from conversion.surface_comparison.cell_registration import register_cells
from conversion.surface_comparison.cmc_classification import classify_congruent_cells


def run_comparison_pipeline(
    reference_map: SurfaceMap,
    comparison_map: SurfaceMap,
    params: ComparisonParams,
) -> ComparisonResult:
    """
    Execute the NIST Congruent Matching Cells (CMC) pipeline.

    1. Per-cell registration via MATLAB-faithful cell_corr_analysis.
    2. CMC classification via median procedure 6 with ESD outlier rejection.

    :param reference_map: The fixed surface map.
    :param comparison_map: The moving surface map.
    :param params: CMC algorithm parameters.
    :returns: ComparisonResult with per-cell results and CMC count.
    """
    result = ComparisonResult()
    result.cells = register_cells(reference_map, comparison_map, params)
    classify_congruent_cells(result, params, reference_map.global_center)
    result.update_summary()
    return result
