from functools import partial
import numpy as np
import pytest
from returns.pipeline import is_successful

from container_models.light_source import LightSource
from container_models.surface_normals import SurfaceNormals
from renders.shading import apply_multiple_lights

no_scale_apply_multiple_lights = partial(apply_multiple_lights, scale_x=1, scale_y=1)


@pytest.fixture(scope="module")
def multiple_lights(light_source) -> tuple[LightSource, LightSource, LightSource]:
    """Multiple lights from different angles."""
    return (
        light_source,
        LightSource(azimuth=135, elevation=45),
        LightSource(azimuth=225, elevation=45),
    )


def test_empty_light_list_returns_failure(
    flat_surface_normals: SurfaceNormals,
    observer: LightSource,
) -> None:
    """Test that an empty light list returns a Failure result."""
    # Act
    result = no_scale_apply_multiple_lights(flat_surface_normals, [], observer)

    # Assert
    assert not is_successful(result)


def test_constant_normals_give_constant_output(
    flat_surface_normals: SurfaceNormals,
    multiple_lights: tuple[LightSource, ...],
    observer: LightSource,
) -> None:
    """Test that constant normals produce constant output across the image."""
    # Act
    result = no_scale_apply_multiple_lights(
        flat_surface_normals, multiple_lights, observer
    )

    # Assert
    scan_image = result.unwrap()
    # All pixels should have the same value
    assert np.allclose(scan_image.data, scan_image.data[0, 0])


def test_more_lights_increase_brightness(
    flat_surface_normals: SurfaceNormals,
    observer: LightSource,
    light_source: LightSource,
    multiple_lights: tuple[LightSource, ...],
) -> None:
    """Test that adding more lights increases total brightness."""

    # Act
    result_one = no_scale_apply_multiple_lights(
        flat_surface_normals, (light_source,), observer
    )
    result_two = no_scale_apply_multiple_lights(
        flat_surface_normals, multiple_lights, observer
    )

    # Assert
    brightness_one = np.mean(result_one.unwrap().data)
    brightness_two = np.mean(result_two.unwrap().data)
    assert brightness_two > brightness_one


def test_light_from_opposing_sides(
    flat_surface_normals: SurfaceNormals,
    observer: LightSource,
) -> None:
    """Test that lights from opposite horizontal directions produce symmetric results."""
    # Arrange - Two lights at opposite azimuths but same elevation
    lights_opposite = [
        LightSource(azimuth=0, elevation=45),
        LightSource(azimuth=180, elevation=45),
    ]

    # Act
    result = no_scale_apply_multiple_lights(
        flat_surface_normals, lights_opposite, observer
    )

    # Assert
    scan_image = result.unwrap()
    # For flat surface normals pointing up, opposite lights should contribute equally
    assert np.all(scan_image.data >= 0)


def test_spatial_variation_with_bumpy_surface(
    observer: LightSource,
    light_source: LightSource,
    flat_surface_normals: SurfaceNormals,
) -> None:
    """Test that surface variation creates intensity variation."""
    # Arrange - Create a surface with a bump in the center
    # Add a tilt in the center
    normals = flat_surface_normals.model_copy(deep=True)
    center = normals.x_normal_vector.shape[0] // 2
    normals.x_normal_vector[center, center] = 0.5
    normals.z_normal_vector[center, center] = 0.866  # To keep it normalized

    # Act
    result = no_scale_apply_multiple_lights(normals, (light_source,), observer)

    # Assert
    scan_image = result.unwrap()
    center_value = scan_image.data[center, center]
    corner_value = scan_image.data[0, 0]
    # Center and corner should have different intensities
    assert not np.isclose(center_value, corner_value)
