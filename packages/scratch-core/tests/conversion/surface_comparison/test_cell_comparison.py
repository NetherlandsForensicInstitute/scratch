import numpy as np
from conversion.surface_comparison.models import SurfaceMap, ComparisonParams
from conversion.surface_comparison.cell_comparison import run_cell_comparison
from skimage.transform import rotate


def test_run_cell_comparison_identity():
    # Arrange
    y, x = np.mgrid[0:100, 0:100]
    data = np.sin(x / 5) * np.cos(y / 5)  # Distinct pattern
    surface = SurfaceMap(data, np.array([1.0, 1.0]), np.array([50.0, 50.0]))
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
    )
    # Act
    results = run_cell_comparison(surface, surface, params)
    # Assertions
    assert len(results) > 0
    for cell in results:
        assert cell.area_cross_correlation_function_score > 0.99
        assert cell.registration_angle == 0.0


def test_run_cell_comparison_rotation_search():
    # Arrange
    y, x = np.mgrid[0:120, 0:120]
    data = np.sin(x / 3) + np.cos(y / 3)
    spacing = np.array([1.0, 1.0])
    reference = SurfaceMap(data, spacing, np.array([60.0, 60.0]))
    target_angle = 1.5
    comparison = SurfaceMap(
        rotate(data, target_angle, preserve_range=True), spacing, np.array([60.0, 60.0])
    )
    params = ComparisonParams(
        cell_size=np.array([40.0, 40.0]),
        search_angle_min=0.0,
        search_angle_max=3.0,
        search_angle_step=0.5,
    )
    # Acts
    results = run_cell_comparison(reference, comparison, params)
    # Assertions
    found_angles = [np.degrees(c.registration_angle) for c in results]
    assert any(np.isclose(angle, target_angle, atol=0.01) for angle in found_angles)
