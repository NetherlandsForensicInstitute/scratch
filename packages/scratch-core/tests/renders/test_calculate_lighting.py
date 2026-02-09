import numpy as np
import pytest

from container_models.base import VectorField
from container_models.light_source import LightSource
from renders.shading import calculate_lighting
from scipy.constants import micro


def test_shape(
    varied_normals: VectorField,
    observer: LightSource,
    light_source: LightSource,
) -> None:
    # Act
    out = calculate_lighting(light_source, observer, varied_normals)
    # Assert
    expected_shape = (
        varied_normals.shape[0],
        varied_normals.shape[1],
    )
    assert out.shape == expected_shape


def test_value_range(
    varied_normals: VectorField,
    observer: LightSource,
    light_source: LightSource,
) -> None:
    # Act
    out = calculate_lighting(light_source, observer, varied_normals)

    # Assert
    assert np.all(out >= 0)
    assert np.all(out <= 1)


def test_constant_normals_give_constant_output(
    varied_normals: VectorField,
    observer: LightSource,
    light_source: LightSource,
) -> None:
    # Act
    out = calculate_lighting(light_source, observer, varied_normals)

    # Assert
    assert np.allclose(out, out[0, 0])


def test_bump_changes_values(
    observer: LightSource,
    light_source: LightSource,
    flat_normals: VectorField,
) -> None:
    """Test that the shader reacts per pixel by giving a bump in the normals."""
    # Arrange
    bump_surface = flat_normals.copy()
    center = flat_normals.shape[0] // 2
    bump_surface[center, center, 2] = 1.3  # Modify z-normal

    # Act
    out = calculate_lighting(light_source, observer, bump_surface)

    # Assert
    center_ = out[center, center]
    border = out[center + 1, center + 1]
    assert not np.allclose(center_, border), (
        "Center pixel should differ from border pixel due to bump."
    )


@pytest.mark.parametrize(
    "light_source,nx,ny,nz",
    [
        pytest.param(
            LightSource(azimuth=0, elevation=0),
            np.ones((10, 10)),
            np.zeros((10, 10)),
            np.zeros((10, 10)),
            id="Light pointing -X, normal pointing +X",
        ),
        pytest.param(
            LightSource(azimuth=180, elevation=0),
            -np.ones((10, 10)),
            np.zeros((10, 10)),
            np.zeros((10, 10)),
            id="Light pointing +X, normal pointing -X",
        ),
        pytest.param(
            LightSource(azimuth=270, elevation=0),
            np.zeros((10, 10)),
            np.ones((10, 10)),
            np.zeros((10, 10)),
            id="Light pointing -Y, normal pointing +Y",
        ),
        pytest.param(
            LightSource(azimuth=90, elevation=0),
            np.zeros((10, 10)),
            -np.ones((10, 10)),
            np.zeros((10, 10)),
            id="Light pointing +Y, normal pointing -Y",
        ),
        pytest.param(
            LightSource(azimuth=0, elevation=90),
            np.zeros((10, 10)),
            np.zeros((10, 10)),
            -np.ones((10, 10)),
            id="Light pointing +Z, normal pointing -Z",
        ),
    ],
)
def test_diffuse_clamps_to_zero(
    light_source: LightSource,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    observer: LightSource,
) -> None:
    """Opposite direction → diffuse should be 0."""
    # Arrange
    normals_scan = np.stack([nx, ny, nz], axis=-1)

    # Act
    out = calculate_lighting(light_source, observer, normals_scan)

    # Assert
    assert np.all(out == 0), "values should be 0."


def test_specular_maximum_case(
    observer: LightSource, flat_normals: VectorField
) -> None:
    """If light, observer, and normal all align, specular should be maximal."""

    # Act
    out = calculate_lighting(observer, observer, flat_normals)

    # Assert
    assert np.allclose(out, 1.0), "(diffuse=1, specular=1), output = (1+1)/2 = 1"


def test_lighting_known_value(
    varied_normals: VectorField,
    observer: LightSource,
    light_source: LightSource,
) -> None:
    expected_constant = 0.04571068

    # Act
    out = calculate_lighting(light_source, observer, varied_normals)

    # Assert
    assert np.allclose(out, expected_constant, atol=micro)
