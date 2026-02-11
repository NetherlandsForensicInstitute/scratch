from pathlib import Path
import pytest

from container_models.image import ImageContainer
import numpy as np
from numpy.testing import assert_array_almost_equal

from mutations.shading import ImageForDisplay
from returns.pipeline import is_successful

from container_models.base import UnitVector
from mutations.shading import LightIntensityMap
from ..helper_function import spherical_to_unit_vector


@pytest.mark.integration
class TestImageForDisplay:
    def test_get_image_for_display_matches_baseline_image(
        self, image_with_nans: ImageContainer, baseline_images_dir: Path
    ):
        get_scan_image_for_display = ImageForDisplay(2)
        # arrange
        verified = np.load(baseline_images_dir / "display_array.npy")
        # act
        display_image = get_scan_image_for_display(image_with_nans).unwrap()
        # assert
        assert_array_almost_equal(display_image.data, verified)


@pytest.fixture(scope="module")
def multiple_lights(
    light_source: UnitVector,
) -> tuple[UnitVector, UnitVector, UnitVector]:
    """Multiple lights from different angles."""
    return (
        light_source,
        spherical_to_unit_vector(azimuth=135, elevation=45),
        spherical_to_unit_vector(azimuth=225, elevation=45),
    )


class TestLightIntensityMap:
    def test_empty_light_list_returns_failure(
        self,
        flat_image: ImageContainer,
        observer: UnitVector,
    ) -> None:
        """Test that an empty light list returns a Failure result."""
        # Arrange
        mutation = LightIntensityMap(light_sources=[], observer=observer)

        # Act
        result = mutation(flat_image.model_copy(deep=True))

        # Assert
        assert not is_successful(result)

    def test_constant_depth_gives_constant_output(
        self,
        flat_image: ImageContainer,
        multiple_lights: tuple[UnitVector, ...],
        observer: UnitVector,
    ) -> None:
        """Test that constant depth (flat surface) produces constant output across the image."""
        # Arrange
        mutation = LightIntensityMap(light_sources=multiple_lights, observer=observer)

        # Act
        result = mutation(flat_image.model_copy(deep=True))

        # Assert
        assert is_successful(result)
        output = result.unwrap()
        # Interior pixels should have the same value (edges have NaN from gradient padding)
        interior = output.data[2:-2, 2:-2]
        assert np.allclose(interior, interior[0, 0], equal_nan=True)

    def test_more_lights_increase_brightness(
        self,
        bumpy_image: ImageContainer,
        observer: UnitVector,
        light_source: UnitVector,
        multiple_lights: tuple[UnitVector, ...],
    ) -> None:
        """Test that adding more lights increases total brightness on a varied surface."""
        # Arrange - use bumpy_image since flat surfaces normalize to same value
        single_light_mutation = LightIntensityMap(
            light_sources=(light_source,), observer=observer
        )
        multi_light_mutation = LightIntensityMap(
            light_sources=multiple_lights, observer=observer
        )

        # Act
        result_one = single_light_mutation(bumpy_image.model_copy(deep=True))
        result_multi = multi_light_mutation(bumpy_image.model_copy(deep=True))

        # Assert
        assert is_successful(result_one)
        assert is_successful(result_multi)
        # With multiple lights from different angles, shadow areas get illuminated
        # resulting in higher minimum brightness (less dark areas)
        min_brightness_one = np.nanmin(result_one.unwrap().data)
        min_brightness_multi = np.nanmin(result_multi.unwrap().data)
        assert min_brightness_multi >= min_brightness_one

    def test_light_from_opposing_sides(
        self,
        flat_image: ImageContainer,
        observer: UnitVector,
    ) -> None:
        """Test that lights from opposite horizontal directions produce valid results."""
        # Arrange - Two lights at opposite azimuths but same elevation
        lights_opposite = [
            spherical_to_unit_vector(azimuth=0, elevation=45),
            spherical_to_unit_vector(azimuth=180, elevation=45),
        ]
        mutation = LightIntensityMap(light_sources=lights_opposite, observer=observer)

        # Act
        result = mutation(flat_image.model_copy(deep=True))

        # Assert
        assert is_successful(result)
        light_intensities = result.unwrap().data
        # For flat surface, all valid pixels should have non-negative intensities
        assert np.all(np.nan_to_num(light_intensities, nan=0.0) >= 0)

    def test_spatial_variation_with_bumpy_surface(
        self,
        bumpy_image: ImageContainer,
        observer: UnitVector,
        light_source: UnitVector,
    ) -> None:
        """Test that surface variation creates intensity variation."""
        # Arrange
        mutation = LightIntensityMap(light_sources=(light_source,), observer=observer)

        # Act
        result = mutation(bumpy_image.model_copy(deep=True))

        # Assert
        assert is_successful(result)
        light_intensities = result.unwrap().data
        center = light_intensities.shape[0] // 2
        center_value = light_intensities[center, center]
        corner_value = light_intensities[2, 2]  # Avoid NaN edges
        # Center and corner should have different intensities due to surface variation
        assert not np.isclose(center_value, corner_value, equal_nan=True)
