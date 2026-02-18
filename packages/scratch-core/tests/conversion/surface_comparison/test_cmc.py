import numpy as np
from conversion.surface_comparison.models import (
    ComparisonResult,
    CellResult,
    ComparisonParams,
)
from conversion.surface_comparison.cmc import classify_cmc_cells


def test_classify_cmc_cells_consensus():
    """Verify that the pairwise voting selects the largest cluster."""
    # Arrange
    cells = [
        CellResult(np.array([0, 0]), np.array([5, 5]), 0.0, 0.8, 1.0),
        CellResult(np.array([10, 10]), np.array([15, 15]), 0.0, 0.8, 1.0),
        CellResult(np.array([20, 20]), np.array([80, 80]), 0.5, 0.8, 1.0),  # Outlier
    ]
    result = ComparisonResult(cells=cells)
    params = ComparisonParams(correlation_threshold=0.5, position_threshold=2.0)
    # Act
    classify_cmc_cells(result, params)
    # Assertions
    assert result.congruent_matching_cells_count == 2
    assert np.allclose(result.consensus_translation, [5.0, 5.0])
