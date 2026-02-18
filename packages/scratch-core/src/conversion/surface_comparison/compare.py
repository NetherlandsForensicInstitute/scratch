import numpy as np
from conversion.surface_comparison.models import (
    SurfaceMap,
    ComparisonResult,
    ComparisonParams,
)
from conversion.surface_comparison.area_comparison import run_area_comparison
from conversion.surface_comparison.cell_comparison import run_cell_comparison
from conversion.surface_comparison.cmc import classify_cmc_cells


def compare_datasets_nist(
    reference_map: SurfaceMap, comparison_map: SurfaceMap, params: ComparisonParams
) -> ComparisonResult:
    """
    Execute the full NIST Congruent Matching Cells pipeline.

    The pipeline follows three major steps:
    1. Global Coarse Alignment: Uses Fourier-Mellin transform to find an initial
       [dx, dy, theta] estimate and eliminate 180-degree ambiguity.
    2. Per-Cell Fine Registration: Divides the reference into a grid and finds
       the best local match for each cell within a localized angular search window.
    3. CMC Classification: Uses a pairwise voting algorithm (Tong et al.) to identify
       cells that share a common registration consensus.

    :param reference_map: The primary (fixed) SurfaceMap.
    :param comparison_map: The secondary (moving) SurfaceMap.
    :param params: Data class containing thresholds and search parameters.
    :returns: ComparisonResult containing similarity metrics and CMC counts.
    """
    result = ComparisonResult()

    # 1. Global Coarse Alignment (Fourier-Mellin)
    # This provides the initial global consensus to center the per-cell search.
    translation, rotation_deg, area_sim = run_area_comparison(
        reference_map, comparison_map
    )
    result.area_similarity = area_sim
    initial_consensus_rad = float(np.radians(rotation_deg))

    # 2. Per-Cell Fine Registration
    # We pass the initial_consensus_rad to narrow the rotation search range
    # and improve computational efficiency/accuracy.
    result.cells = run_cell_comparison(
        reference_map, comparison_map, params, initial_rotation=initial_consensus_rad
    )

    # 3. CMC Classification (Pairwise Voting)
    # This identifies which cells are "congruent" based on the NIST criteria.
    classify_cmc_cells(result, params)

    # update_summary() is called inside classify_cmc_cells, but we ensure final state consistency here.
    result.update_summary()

    return result
