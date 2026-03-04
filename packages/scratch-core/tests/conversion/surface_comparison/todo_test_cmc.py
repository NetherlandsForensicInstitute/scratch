import numpy as np

from conversion.surface_comparison.cmc_classification import classify_congruent_cells
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import (
    Cell,
    ComparisonParams,
    ComparisonResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surface_map(
    height_map: np.ndarray, pixel_spacing_m: float = 1e-6
) -> ScanImage:
    scale = pixel_spacing_m
    return ScanImage(data=height_map, scale_x=scale, scale_y=scale)


# ---------------------------------------------------------------------------
# classify_congruent_cells
# ---------------------------------------------------------------------------


def test_classify_cmc_cells_consensus():
    """Two inlier cells and one positional outlier: only the inliers become CMC.

    All registration angles are close to zero, so the consensus angle is 0 and
    _rotate_points is the identity.  Position residuals are [5,5], [5,5], [60,60];
    their median translation is [5,5].  Cell 2's position error [55,55] exceeds
    position_threshold=2, so it is not classified as a CMC.

    Cell 2 also has angle_reference=0.5 rad, which is far outside the default
    angle_threshold of 2 degrees, providing a second independent reason for
    rejection.
    """
    cells = [
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([0.0, 0.0]),
            center_comparison=np.array([5.0, 5.0]),
            angle_reference=0.0,
            best_score=0.8,
            fill_fraction_reference=1.0,
        ),
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([10.0, 10.0]),
            center_comparison=np.array([15.0, 15.0]),
            angle_reference=0.0,
            best_score=0.8,
            fill_fraction_reference=1.0,
        ),
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([20.0, 20.0]),
            center_comparison=np.array([80.0, 80.0]),
            angle_reference=0.5 * 180 / np.pi,
            best_score=0.8,
            fill_fraction_reference=1.0,
        ),
    ]
    result = ComparisonResult(cells=cells)
    params = ComparisonParams(correlation_threshold=0.5, position_threshold=2.0)
    reference_center = np.array([0.0, 0.0])

    classify_congruent_cells(result, params, reference_center)

    assert result.congruent_matching_cells_count == 2
    assert result.cells[0].is_congruent is True
    assert result.cells[1].is_congruent is True
    assert result.cells[2].is_congruent is False
    assert np.allclose(result.consensus_translation, [5.0, 5.0])
    # consensus_rotation is degrees — add if you want to check it:
    assert np.isclose(result.consensus_rotation, 0.0, atol=0.01)


def test_classify_cmc_score_below_threshold():
    """A cell whose ACCF score is below correlation_threshold is not CMC,
    even when its position and angle agree perfectly with the consensus."""
    cells = [
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([0.0, 0.0]),
            center_comparison=np.array([5.0, 5.0]),
            angle_reference=0.0,
            best_score=0.8,
            fill_fraction_reference=1.0,
        ),
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([10.0, 10.0]),
            center_comparison=np.array([15.0, 5.0]),
            angle_reference=0.0,
            best_score=0.3,
            fill_fraction_reference=1.0,
        ),
    ]
    result = ComparisonResult(cells=cells)
    params = ComparisonParams(correlation_threshold=0.5, position_threshold=50.0)
    reference_center = np.array([0.0, 0.0])

    classify_congruent_cells(result, params, reference_center)

    assert result.cells[0].is_congruent is True
    assert result.cells[1].is_congruent is False


def test_classify_cmc_empty_cells():
    """Empty cell list: function returns without error and count stays zero."""
    result = ComparisonResult(cells=[])
    params = ComparisonParams()
    reference_center = np.array([0.0, 0.0])

    classify_congruent_cells(result, params, reference_center)

    assert result.congruent_matching_cells_count == 0


def test_classify_cmc_all_congruent():
    """All four cells agree perfectly on translation: all become CMC."""
    cells = [
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([0.0, 0.0]),
            center_comparison=np.array([5.0, 5.0]),
            angle_reference=0.0,
            best_score=0.9,
            fill_fraction_reference=1.0,
        ),
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([10.0, 0.0]),
            center_comparison=np.array([15.0, 5.0]),
            angle_reference=0.0,
            best_score=0.9,
            fill_fraction_reference=1.0,
        ),
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([0.0, 10.0]),
            center_comparison=np.array([5.0, 15.0]),
            angle_reference=0.0,
            best_score=0.9,
            fill_fraction_reference=1.0,
        ),
        Cell(
            cell_data=np.array([[0.0, 0.0], [0.0, 0.0]]),
            center_reference=np.array([10.0, 10.0]),
            center_comparison=np.array([15.0, 15.0]),
            angle_reference=0.0,
            best_score=0.9,
            fill_fraction_reference=1.0,
        ),
    ]
    result = ComparisonResult(cells=cells)
    params = ComparisonParams(correlation_threshold=0.5, position_threshold=1.0)
    reference_center = np.array([5.0, 5.0])

    classify_congruent_cells(result, params, reference_center)

    assert result.congruent_matching_cells_count == 4
    assert all(c.is_congruent for c in result.cells)
