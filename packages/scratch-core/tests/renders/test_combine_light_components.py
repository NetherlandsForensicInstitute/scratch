import numpy as np
import pytest
from scipy.constants import micro

from container_models.base import UnitVector, VectorField
from renders.shading import combine_light_components

from ..helper_function import spherical_to_unit_vector

TEST_IMAGE_SIZE = 10


def test_shape(
    flat_normals: VectorField,
    observer: UnitVector,
    light_source: UnitVector,
) -> None:
    """Test that output shape matches input spatial dimensions."""
    # Act
    out = combine_light_components(flat_normals, light_source, observer)

    # Assert
    assert out.shape == (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)


def test_value_range(
    flat_normals: VectorField,
    observer: UnitVector,
    light_source: UnitVector,
) -> None:
    """Test that output values are in valid range [0, 1]."""
    # Act
    out = combine_light_components(flat_normals, light_source, observer)

    # Assert
    assert np.all(out >= 0)
    assert np.all(out <= 1)


def test_constant_normals_give_constant_output(
    flat_normals: VectorField,
    observer: UnitVector,
    light_source: UnitVector,
) -> None:
    """Test that constant normals produce constant output."""
    # Act
    out = combine_light_components(flat_normals, light_source, observer)

    # Assert
    assert np.allclose(out, out[0, 0])


def test_bump_changes_values(
    flat_normals: VectorField,
    observer: UnitVector,
    light_source: UnitVector,
) -> None:
    """Test that the shader reacts per pixel by giving a bump in the normals."""
    # Arrange
    bump_surface = flat_normals.copy()
    center = flat_normals.shape[0] // 2
    # Modify normal at center to point differently (normalize to keep unit length)
    bump_surface[center, center] = np.array([0.5, 0.0, 0.866])

    # Act
    out = combine_light_components(bump_surface, light_source, observer)

    # Assert
    center_value = out[center, center]
    border_value = out[center + 1, center + 1]
    assert not np.isclose(center_value, border_value), (
        "Center pixel should differ from border pixel due to bump."
    )


@pytest.mark.parametrize(
    "light,nx,ny,nz",
    [
        pytest.param(
            spherical_to_unit_vector(azimuth=270, elevation=0),
            np.zeros((10, 10)),
            np.ones((10, 10)),
            np.zeros((10, 10)),
            id="Light pointing -Y, normal pointing +Y",
        ),
        pytest.param(
            spherical_to_unit_vector(azimuth=90, elevation=0),
            np.zeros((10, 10)),
            -np.ones((10, 10)),
            np.zeros((10, 10)),
            id="Light pointing +Y, normal pointing -Y",
        ),
        pytest.param(
            spherical_to_unit_vector(azimuth=0, elevation=90),
            np.zeros((10, 10)),
            np.zeros((10, 10)),
            -np.ones((10, 10)),
            id="Light pointing +Z, normal pointing -Z",
        ),
    ],
)
def test_diffuse_clamps_to_zero(
    light: UnitVector,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    observer: UnitVector,
) -> None:
    """Opposite direction → combined output should be 0 (diffuse=0, specular=0)."""
    # Arrange
    normals = np.stack([nx, ny, nz], axis=-1)

    # Act
    out = combine_light_components(normals, light, observer)

    # Assert - when light and normal are opposite and observer is +Z,
    # both diffuse and specular should be 0
    assert np.all(out == 0), "Values should be 0 when light and normal are opposite."


def test_specular_maximum_case(
    flat_normals: VectorField,
    observer: UnitVector,
) -> None:
    """If light, observer, and normal all align, specular should be maximal."""
    # Act - use observer as light direction (both pointing +Z, normals pointing +Z)
    out = combine_light_components(flat_normals, observer, observer)

    # Assert
    assert np.allclose(out, 1.0), "(diffuse=1, specular=1), output = (1+1)/2 = 1"


def test_lighting_known_value(
    flat_normals: VectorField,
    observer: UnitVector,
    light_source: UnitVector,
) -> None:
    """Test against a known expected value for consistency."""
    # For flat normals (0,0,1) with light at azimuth=45, elevation=45 and observer at (0,0,1)
    expected_constant = 0.47855339

    # Act
    out = combine_light_components(flat_normals, light_source, observer)

    # Assert
    assert np.allclose(out, expected_constant, atol=micro)
