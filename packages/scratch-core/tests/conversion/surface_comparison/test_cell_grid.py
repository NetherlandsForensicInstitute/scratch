import numpy as np
from conversion.surface_comparison.models import SurfaceMap, ComparisonParams
from conversion.surface_comparison.cell_grid import (
    find_optimal_cell_origin,
    generate_cell_centers,
)


def test_find_optimal_cell_origin():
    """Verify that the optimizer returns a physical coordinate."""
    # Arrange
    data = np.ones((100, 100))
    surface = SurfaceMap(data, np.array([1.0, 1.0]), np.array([50.0, 50.0]))
    params = ComparisonParams(cell_size=np.array([20.0, 20.0]))
    # Act
    origin = find_optimal_cell_origin(surface, params)
    # Assertions
    assert origin.shape == (2,)
    assert np.all(origin >= 0)


def test_generate_cell_centers():
    """Ensure centers are within surface bounds."""
    # Arrange
    data = np.ones((100, 100))
    surface = SurfaceMap(data, np.array([1.0, 1.0]), np.array([50.0, 50.0]))
    params = ComparisonParams(cell_size=np.array([25.0, 25.0]))
    # Act
    centers = generate_cell_centers(surface, np.array([0.0, 0.0]), params)
    # Assertions
    assert len(centers) == 16  # 4x4 grid
    assert np.all(centers[:, 0] < 100.0)
