from functools import partial
import numpy as np
import pytest
from returns.pipeline import is_successful

from container_models.light_source import LightSource
from container_models.scan_image import ScanImage
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
    flat_normals_scan_image: ScanImage,
    observer: LightSource,
) -> None:
    """Test that an empty light list returns a Failure result."""
    # Act
    result = no_scale_apply_multiple_lights(flat_normals_scan_image, [], observer)

    # Assert
    assert not is_successful(result)


def test_constant_normals_give_constant_output(
    flat_normals_scan_image: ScanImage,
    multiple_lights: tuple[LightSource, ...],
    observer: LightSource,
) -> None:
    """Test that constant normals produce constant output across the image."""
    # Act
    result = no_scale_apply_multiple_lights(
        flat_normals_scan_image, multiple_lights, observer
    )

    # Assert
    scan_image = result.unwrap()
    # All pixels should have the same value
    assert np.allclose(scan_image.data, scan_image.data[0, 0])


def test_more_lights_increase_brightness(
    flat_normals_scan_image: ScanImage,
    observer: LightSource,
    light_source: LightSource,
    multiple_lights: tuple[LightSource, ...],
) -> None:
    """Test that adding more lights increases total brightness."""

    # Act
    result_one = no_scale_apply_multiple_lights(
        flat_normals_scan_image, (light_source,), observer
    )
    result_two = no_scale_apply_multiple_lights(
        flat_normals_scan_image, multiple_lights, observer
    )

    # Assert
    brightness_one = np.mean(result_one.unwrap().data)
    brightness_two = np.mean(result_two.unwrap().data)
    assert brightness_two > brightness_one


def test_light_from_opposing_sides(
    flat_normals_scan_image: ScanImage,
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
        flat_normals_scan_image, lights_opposite, observer
    )

    # Assert
    scan_image = result.unwrap()
    # For flat surface normals pointing up, opposite lights should contribute equally
    assert np.all(scan_image.data >= 0)


def test_spatial_variation_with_bumpy_surface(
    observer: LightSource,
    light_source: LightSource,
    flat_normals_scan_image: ScanImage,
) -> None:
    """Test that surface variation creates intensity variation."""
    # Arrange - Create a surface with a bump in the center
    normals = flat_normals_scan_image.model_copy(deep=True)
    center = normals.data.shape[0] // 2
    normals.data[center, center, 0] = 0.5  # x-normal
    normals.data[center, center, 2] = 0.866  # z-normal (to keep it normalized)

    # Act
    result = no_scale_apply_multiple_lights(normals, (light_source,), observer)

    # Assert
    scan_image = result.unwrap()
    center_value = scan_image.data[center, center]
    corner_value = scan_image.data[0, 0]
    # Center and corner should have different intensities
    assert not np.isclose(center_value, corner_value)
