import numpy as np

from image_generation.translations import apply_multiple_lights
from utils.array_definitions import ScanMap2DArray


class TestMultipleLights:
    IMAGE_HEIGHT = 10
    IMAGE_WIDTH = 12

    def _simple_calc(
        self,
        light_vector,
        observer_vector,
        surface_normals,
        specular_factor: float = 1.0,
        phong_exponent: int = 1,
    ) -> ScanMap2DArray:
        """A dumb calculator that returns a constant equal to the x-component of the light."""
        return np.sum(surface_normals * light_vector, axis=-1)

    def test_apply_multiple_lights(self) -> None:
        """test if for each light a layer is calculated with the given function."""
        # Arrange
        light1 = np.array([1.0, 0.5, 1.0])
        light2 = np.array([0.5, 1.0, 0.9])
        light3 = np.array([0.5, 0.5, 0.5])
        lights_combined = (light1, light2, light3)
        observer_vector = np.array([0.0, 1.0, 0.0])
        nx = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        ny = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        nz = np.ones((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

        base_images = np.stack([nx, ny, nz], axis=-1)
        expected_result_light_1 = self._simple_calc(
            light1, observer_vector, base_images
        )
        expected_result_light_2 = self._simple_calc(
            light2, observer_vector, base_images
        )
        expected_result_light_3 = self._simple_calc(
            light3, observer_vector, base_images
        )

        # Act
        result = apply_multiple_lights(
            surface_normals=base_images,
            light_vectors=lights_combined,
            observer_vector=observer_vector,
            lighting_calculator=self._simple_calc,
        )

        # Assert
        assert result.shape == (
            self.IMAGE_WIDTH,
            self.IMAGE_HEIGHT,
            len(lights_combined),
        )
        assert np.all(result[..., 0] == expected_result_light_1)
        assert np.all(result[..., 1] == expected_result_light_2)
        assert np.all(result[..., 2] == expected_result_light_3)
