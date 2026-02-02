from functools import partial
import numpy as np
import pytest
from scipy.constants import milli

from container_models import ImageContainer
from renders import compute_surface_normals
from container_models.base import BinaryMask, VectorField

IMAGE_SIZE = 20
BUMP_SIZE = 6
BUMP_HEIGHT = 4
BUMP_CENTER = IMAGE_SIZE // 2
BUMP_SLICE = slice(BUMP_CENTER - BUMP_SIZE // 2, BUMP_CENTER + BUMP_SIZE // 2)
NoScaleImageContainer = partial(ImageContainer, scale_x=1, scale_y=1)


@pytest.fixture
def inner_mask() -> BinaryMask:
    """Mask of all pixels except the 1-pixel border."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


@pytest.fixture
def outer_mask(inner_mask: BinaryMask) -> BinaryMask:
    """Inverse of inner_mask: the NaN border."""
    return ~inner_mask


def are_normals_allclose(
    normals: VectorField,
    mask: BinaryMask,
    expected: tuple[float, float, float],
) -> bool:
    """Assert nx, ny, nz at mask match expected 3-tuple."""
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    return (
        np.allclose(nx[mask], expected[0], atol=milli)
        and np.allclose(ny[mask], expected[1], atol=milli)
        and np.allclose(nz[mask], expected[2], atol=milli)
    )


def is_all_nan(normals: VectorField, mask: BinaryMask) -> np.bool_:
    """All channels must be NaN within mask."""
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    return (
        np.isnan(nx[mask]).all()
        and np.isnan(ny[mask]).all()
        and np.isnan(nz[mask]).all()
    )


def has_nan(normals: VectorField, mask: BinaryMask) -> np.bool_:
    """No channel should contain NaN within mask."""
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    return (
        np.isnan(nx[mask]).any()
        and np.isnan(ny[mask]).any()
        and np.isnan(nz[mask]).any()
    )


@pytest.fixture(scope="module")
def flat_neutral_image() -> ImageContainer:
    return NoScaleImageContainer(data=np.zeros((IMAGE_SIZE, IMAGE_SIZE)))


def test_slope_has_nan_border(
    inner_mask: BinaryMask, outer_mask: BinaryMask, flat_neutral_image: ImageContainer
) -> None:
    """
    The image is 1 pixel smaller on all sides due to the slope calculation.
    This is filled with NaN values to get the same shape as original image
    """
    # Act
    surface_normals = compute_surface_normals(flat_neutral_image).unwrap()

    # Assert
    assert not has_nan(surface_normals, inner_mask)
    assert is_all_nan(surface_normals, outer_mask)


def test_flat_surface_returns_flat_surface(
    inner_mask: BinaryMask, flat_neutral_image: ImageContainer
) -> None:
    """Given a flat surface the depth map should also be flat."""

    # Act
    surface_normals = compute_surface_normals(flat_neutral_image).unwrap()

    # Assert
    assert are_normals_allclose(surface_normals, inner_mask, (0, 0, 1))


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
def test_linear_slope(step_x: float, step_y: float, inner_mask: BinaryMask) -> None:
    """Test linear slopes in X, Y, or both directions."""
    # Arrange
    x_vals = np.arange(IMAGE_SIZE) * step_x
    y_vals = np.arange(IMAGE_SIZE) * step_y
    input_image = ImageContainer(
        data=y_vals[:, None] + x_vals[None, :], scale_x=1, scale_y=1
    )
    norm = np.sqrt(step_x**2 + step_y**2 + 1)
    expected = (-step_x / norm, step_y / norm, 1 / norm)

    # Act
    surface_normals = compute_surface_normals(input_image).unwrap()

    # Assert
    assert are_normals_allclose(surface_normals, inner_mask, expected)


@pytest.fixture
def image_with_bump() -> ImageContainer:
    data = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
    data[BUMP_SLICE, BUMP_SLICE] = BUMP_HEIGHT
    return NoScaleImageContainer(data=data)


def test_location_slope_is_where_expected(
    inner_mask: BinaryMask,
    image_with_bump: ImageContainer,
) -> None:
    """Check that slope calculation is localized to the bump coordination an offset of 1 is used for the slope."""
    # Arrange
    bump_mask = np.zeros_like(image_with_bump.data, dtype=bool)
    bump_mask[
        BUMP_SLICE.start - 1 : BUMP_SLICE.stop + 1,
        BUMP_SLICE.start - 1 : BUMP_SLICE.stop + 1,
    ] = True
    outside_bump_mask = ~bump_mask & inner_mask

    # Act
    surface_normals = compute_surface_normals(image_with_bump).unwrap()
    nx, ny, nz = (
        surface_normals[..., 0],
        surface_normals[..., 1],
        surface_normals[..., 2],
    )

    # Assert
    assert np.any(np.abs(nx[bump_mask]) > 0), "nx should have slope inside bump"
    assert np.any(np.abs(ny[bump_mask]) > 0), "ny should have slope inside bump"
    assert np.any(np.abs(nz[bump_mask]) != 1), "nz should deviate from 1 inside bump"

    assert are_normals_allclose(surface_normals, outside_bump_mask, (0, 0, 1))


def test_corner_of_slope(image_with_bump: ImageContainer) -> None:
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
    surface_normals = compute_surface_normals(image_with_bump).unwrap()
    nz = surface_normals[..., 2]

    # Assert
    assert nz[corner[0], corner[1]] == expected_corner_value, (
        "corner of x and y should have unit normal of x and y"
    )
