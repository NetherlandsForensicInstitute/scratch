import numpy as np
import pytest
from numpy.testing import assert_allclose

from image_generation.translations import compute_surface_normals
from utils.array_definitions import ScanVectorField2DArray, ScanMap2DArray, MaskArray


TOLERANCE = 1e-3
IMAGE_SIZE = 20
BUMP_SIZE = 6
BUMP_HEIGHT = 4
BUMP_CENTER = IMAGE_SIZE // 2
BUMP_SLICE = slice(BUMP_CENTER - BUMP_SIZE // 2, BUMP_CENTER + BUMP_SIZE // 2)


@pytest.fixture
def inner_mask() -> MaskArray:
    """Mask of all pixels except the 1-pixel border."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


@pytest.fixture
def outer_mask(inner_mask: MaskArray) -> MaskArray:
    """Inverse of inner_mask: the NaN border."""
    return ~inner_mask


def assert_normals_close(
    normals: ScanVectorField2DArray,
    mask: MaskArray,
    expected: tuple[float, float, float],
    atol=1e-3,
) -> None:
    """Assert nx, ny, nz at mask match expected 3-tuple."""
    for component, expected_value in zip(np.moveaxis(normals, -1, 0), expected):
        np.testing.assert_allclose(component[mask], expected_value, atol=atol)


def assert_all_nan(normals: ScanMap2DArray, mask: MaskArray) -> None:
    """All channels must be NaN within mask."""
    assert np.isnan(normals[mask]).all()


def assert_no_nan(normals: ScanMap2DArray, mask: MaskArray) -> None:
    """No channel should contain NaN within mask."""
    assert ~np.isnan(normals[mask]).any()


def test_slope_has_nan_border(
    inner_mask: MaskArray,
    outer_mask: MaskArray,
) -> None:
    """
    The image is 1 pixel smaller on all sides due to the slope calculation.
    This is filled with NaN values to get the same shape as original image
    """
    # Arrange
    input_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    # Act
    surface_normals = compute_surface_normals(
        depth_data=input_image, x_dimension=1, y_dimension=1
    )

    # Assert
    assert_no_nan(surface_normals, inner_mask)
    assert_all_nan(surface_normals, outer_mask)


def test_flat_surface_returns_flat_surface(
    inner_mask: MaskArray,
    outer_mask: MaskArray,
) -> None:
    """Given a flat surface the depth map should also be flat."""
    # Arrange
    input_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    # Act
    surface_normals = compute_surface_normals(
        depth_data=input_image, x_dimension=1, y_dimension=1
    )

    # Assert
    assert_normals_close(surface_normals, inner_mask, (0, 0, 1), atol=TOLERANCE)


@pytest.mark.parametrize(
    "step_x, step_y",
    [
        pytest.param(2.0, 0.0, id="step increase in x"),
        pytest.param(0.0, 2.0, id="step increase in y"),
        pytest.param(2.0, 2.0, id="step increase in x and y"),
        pytest.param(2.0, -2.0, id="positive and negative steps"),
        pytest.param(-2.0, -2.0, id="negative x and y steps"),
    ],
)
def test_linear_slope(step_x: float, step_y: float, inner_mask: MaskArray) -> None:
    """Test linear slopes in X, Y, or both directions."""
    # Arrange
    x_vals = np.arange(IMAGE_SIZE) * step_x
    y_vals = np.arange(IMAGE_SIZE) * step_y
    input_image = y_vals[:, None] + x_vals[None, :]
    norm = np.sqrt(step_x**2 + step_y**2 + 1)
    expected = (-step_x / norm, step_y / norm, 1 / norm)

    # Act
    surface_normals = compute_surface_normals(input_image, 1, 1)

    # Assert
    assert_normals_close(surface_normals, inner_mask, expected, atol=TOLERANCE)


@pytest.fixture
def input_depth_map_with_bump() -> ScanMap2DArray:
    input_depth_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
    input_depth_map[BUMP_SLICE, BUMP_SLICE] = BUMP_HEIGHT
    return input_depth_map


def test_location_slope_is_where_expected(
    inner_mask: MaskArray,
    input_depth_map_with_bump: ScanMap2DArray,
) -> None:
    """Check that slope calculation is localized to the bump coordination an offset of 1 is used for the slope."""
    # Arrange
    bump_mask = np.zeros_like(input_depth_map_with_bump, dtype=bool)
    bump_mask[
        BUMP_SLICE.start - 1 : BUMP_SLICE.stop + 1,
        BUMP_SLICE.start - 1 : BUMP_SLICE.stop + 1,
    ] = True
    outside_bump_mask = ~bump_mask & inner_mask

    # Act
    surface_normals = compute_surface_normals(
        depth_data=input_depth_map_with_bump, x_dimension=1, y_dimension=1
    )

    # Assert
    assert np.any(np.abs(surface_normals[..., 0][bump_mask]) > 0), (
        "nx should have slope inside bump"
    )
    assert np.any(np.abs(surface_normals[..., 1][bump_mask]) > 0), (
        "ny should have slope inside bump"
    )
    assert np.any(np.abs(surface_normals[..., 2][bump_mask]) != 1), (
        "nz should deviate from 1 inside bump"
    )

    assert_normals_close(surface_normals, outside_bump_mask, (0, 0, 1), atol=TOLERANCE)


def test_corner_of_slope(
    inner_mask: MaskArray,
    input_depth_map_with_bump: ScanMap2DArray,
) -> None:
    """Test if the corner of the slope is an extension of x, y"""
    # Arrange
    corner = (
        BUMP_CENTER - BUMP_SIZE // 2,
        BUMP_CENTER - BUMP_SIZE // 2,
    )
    expected_corner_value = 1 / np.sqrt(
        (BUMP_HEIGHT // 2) ** 2 + (BUMP_HEIGHT // 2) ** 2 + 1
    )

    # Act
    surface_normals = compute_surface_normals(input_depth_map_with_bump, 1, 1)

    # Assert

    assert_allclose(
        surface_normals[corner[0], corner[1], 2],
        expected_corner_value,
        atol=TOLERANCE,
        err_msg="corner of x and y should have unit normal of x and y",
    )
