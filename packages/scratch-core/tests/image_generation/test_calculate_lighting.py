import numpy as np
import pytest

from image_generation.data_formats import LightSource
from image_generation.translations import calculate_lighting
from utils.array_definitions import (
    UnitVector3DArray,
    ScanVectorField2DArray,
    ScanMap2DArray,
)


class TestCalculateLighting:
    TEST_IMAGE_WIDTH = 10
    TEST_IMAGE_HEIGHT = 10
    TOLERANCE = 1e-5

    @pytest.fixture(scope="class")
    def light_vector(self) -> UnitVector3DArray:
        return LightSource(azimuth=45, elevation=45).unit_vector

    @pytest.fixture(scope="class")
    def observer_vector(self) -> UnitVector3DArray:
        return LightSource(azimuth=0, elevation=90).unit_vector

    @pytest.fixture(scope="class")
    def base_images(self) -> ScanVectorField2DArray:
        nx = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.7)
        ny = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.6)
        nz = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.2)
        return np.stack([nx, ny, nz], axis=-1)

    def test_shape(
        self,
        base_images: ScanVectorField2DArray,
        observer_vector: UnitVector3DArray,
        light_vector: UnitVector3DArray,
    ) -> None:
        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert out.shape == (self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT)

    def test_value_range(
        self,
        base_images: ScanVectorField2DArray,
        observer_vector: UnitVector3DArray,
        light_vector: UnitVector3DArray,
    ) -> None:
        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert np.all(out >= 0)
        assert np.all(out <= 1)

    def test_constant_normals_give_constant_output(
        self,
        base_images: ScanVectorField2DArray,
        observer_vector: UnitVector3DArray,
        light_vector: UnitVector3DArray,
    ) -> None:
        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert np.allclose(out, out[0, 0])

    def test_bump_changes_values(
        self, observer_vector: UnitVector3DArray, light_vector: UnitVector3DArray
    ) -> None:
        """Test that the shader reacts per pixel by giving a bump in the normals. and thest the location is changed"""
        # Arrange
        nx = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        ny = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        nz = np.ones((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        nz[self.TEST_IMAGE_WIDTH // 2, self.TEST_IMAGE_HEIGHT // 2] = 1.3
        base_images = np.stack([nx, ny, nz], axis=-1)

        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        center = out[self.TEST_IMAGE_WIDTH // 2, self.TEST_IMAGE_HEIGHT // 2]
        border = out[
            (self.TEST_IMAGE_WIDTH // 2) + 1, (self.TEST_IMAGE_HEIGHT // 2) + 1
        ]
        assert not np.allclose(center, border), (
            "Center pixel should differ from border pixel due to bump."
        )

    @pytest.mark.parametrize(
        "light_source,nx,ny,nz",
        [
            pytest.param(
                np.array([-1.0, 0.0, 0.0]),
                np.ones((10, 10)),
                np.zeros((10, 10)),
                np.zeros((10, 10)),
                id="Light pointing -X, normal pointing +X",
            ),
            pytest.param(
                np.array([1.0, 0.0, 0.0]),
                -np.ones((10, 10)),
                np.zeros((10, 10)),
                np.zeros((10, 10)),
                id="Light pointing +X, normal pointing -X",
            ),
            pytest.param(
                np.array([0.0, -1.0, 0.0]),
                np.zeros((10, 10)),
                np.ones((10, 10)),
                np.zeros((10, 10)),
                id="Light pointing -Y, normal pointing +Y",
            ),
            pytest.param(
                np.array([0.0, 1.0, 0.0]),
                np.zeros((10, 10)),
                -np.ones((10, 10)),
                np.zeros((10, 10)),
                id="Light pointing +Y, normal pointing -Y",
            ),
            pytest.param(
                np.array([0.0, 0.0, 1.0]),
                np.zeros((10, 10)),
                np.zeros((10, 10)),
                -np.ones((10, 10)),
                id="Light pointing +Z, normal pointing -Z",
            ),
        ],
    )
    def test_diffuse_clamps_to_zero(
        self,
        light_source: UnitVector3DArray,
        nx: ScanMap2DArray,
        ny: ScanMap2DArray,
        nz: ScanMap2DArray,
        observer_vector: UnitVector3DArray,
    ) -> None:
        """Opposite direction → diffuse should be 0."""
        # Arrange
        base_images = np.stack([nx, ny, nz], axis=-1)
        # Act
        out = calculate_lighting(light_source, observer_vector, base_images)

        # Assert
        assert np.all(out == 0), "values should be 0."

    def test_specular_maximum_case(self, observer_vector: UnitVector3DArray) -> None:
        """If light, observer, and normal all align, specular should be maximal."""
        # Arrange
        nx = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        ny = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        nz = np.ones((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        base_images = np.stack([nx, ny, nz], axis=-1)
        light_vector = np.array([0.0, 0.0, 1.0])

        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert np.allclose(out, 1.0), "(diffuse=1, specular=1), output = (1+1)/2 = 1"

    def test_lighting_known_value(
        self,
        base_images: ScanMap2DArray,
        observer_vector: UnitVector3DArray,
        light_vector: UnitVector3DArray,
    ) -> None:
        expected_constant = 0.04571068

        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert np.allclose(out, expected_constant, atol=self.TOLERANCE)
