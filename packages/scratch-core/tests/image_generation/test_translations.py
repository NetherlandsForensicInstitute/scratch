import numpy as np
import pytest
from numpy._typing import NDArray
from numpy.testing import assert_allclose

from image_generation.data_formats import LightAngle
from image_generation.translations import calculate_lighting
from image_generation.translations import (
    compute_surface_normals,
    normalize_intensity_map,
    apply_multiple_lights,
)
from utils.array_definitions import IMAGE_3_LAYER_STACK_ARRAY, NORMAL_VECTOR


class TestComputeSurfaceNormals:
    TOLERANCE = 1e-6
    IMAGE_SIZE = 20
    BUMP_SIZE = 6
    BUMP_HEIGHT = 4
    BUMP_CENTER = IMAGE_SIZE // 2
    BUMP_SLICE = slice(BUMP_CENTER - BUMP_SIZE // 2, BUMP_CENTER + BUMP_SIZE // 2)

    @pytest.fixture
    def inner_mask(self) -> NDArray[tuple[bool, bool]]:
        """Mask of all pixels except the 1-pixel border."""
        mask = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=bool)
        mask[1:-1, 1:-1] = True
        return mask

    @pytest.fixture
    def outer_mask(
        self, inner_mask: NDArray[tuple[bool, bool]]
    ) -> NDArray[tuple[bool, bool]]:
        """Inverse of inner_mask: the NaN border."""
        return ~inner_mask

    def assert_normals_close(self, normals, mask, expected, atol=1e-6):
        """Assert nx, ny, nz at mask match expected 3-tuple."""
        nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
        exp_x, exp_y, exp_z = expected
        np.testing.assert_allclose(nx[mask], exp_x, atol=atol)
        np.testing.assert_allclose(ny[mask], exp_y, atol=atol)
        np.testing.assert_allclose(nz[mask], exp_z, atol=atol)

    def assert_all_nan(self, normals, mask):
        """All channels must be NaN within mask."""
        assert np.isnan(normals[mask]).all()

    def assert_no_nan(self, normals, mask):
        """No channel should contain NaN within mask."""
        assert ~np.isnan(normals[mask]).any()

    def test_slope_has_nan_border(
        self,
        inner_mask: NDArray[tuple[bool, bool]],
        outer_mask: NDArray[tuple[bool, bool]],
    ) -> None:
        """The image is 1 pixel smaller on all sides due to the slope calculation.
        This is filled with NaN values to get the same shape as original image"""
        # Arrange
        input_image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))

        # Act
        surface_normals = compute_surface_normals(
            depth_data=input_image, x_dimension=1, y_dimension=1
        )

        # Assert
        self.assert_no_nan(surface_normals, inner_mask)
        self.assert_all_nan(surface_normals, outer_mask)

    def test_flat_surface_returns_flat_surface(
        self,
        inner_mask: NDArray[tuple[bool, bool]],
        outer_mask: NDArray[tuple[bool, bool]],
    ) -> None:
        """Given a flat surface the depth map should also be flat."""
        # Arrange
        input_image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))

        # Act
        surface_normals = compute_surface_normals(
            depth_data=input_image, x_dimension=1, y_dimension=1
        )

        # Assert
        self.assert_normals_close(
            surface_normals, inner_mask, (0, 0, 1), atol=self.TOLERANCE
        )

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
    def test_linear_slope(
        self, step_x: int, step_y: int, inner_mask: NDArray[tuple[bool, bool]]
    ) -> None:
        """Test linear slopes in X, Y, or both directions."""
        # Arrange
        x_vals = np.arange(self.IMAGE_SIZE) * step_x
        y_vals = np.arange(self.IMAGE_SIZE) * step_y
        input_image = y_vals[:, None] + x_vals[None, :]
        norm = np.sqrt(step_x**2 + step_y**2 + 1)
        expected = (-step_x / norm, step_y / norm, 1 / norm)

        # Act
        surface_normals = compute_surface_normals(input_image, 1, 1)

        # Assert
        self.assert_normals_close(
            surface_normals, inner_mask, expected, atol=self.TOLERANCE
        )

    @pytest.fixture
    def input_depth_map_with_bump(
        self,
    ) -> NDArray[tuple[int, int]]:
        input_depth_map = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=int)
        input_depth_map[self.BUMP_SLICE, self.BUMP_SLICE] = self.BUMP_HEIGHT
        return input_depth_map

    def test_location_slope_is_where_expected(
        self,
        inner_mask: NDArray[tuple[bool, bool]],
        input_depth_map_with_bump: NDArray[tuple[int, int]],
    ) -> None:
        """Check that slope calculation is localized to the bump coordination an offset of 1 is used for the slope."""
        # Arrange
        bump_mask = np.zeros_like(input_depth_map_with_bump, dtype=bool)
        bump_mask[
            self.BUMP_SLICE.start - 1 : self.BUMP_SLICE.stop + 1,
            self.BUMP_SLICE.start - 1 : self.BUMP_SLICE.stop + 1,
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

        self.assert_normals_close(
            surface_normals, outside_bump_mask, (0, 0, 1), atol=self.TOLERANCE
        )

    def test_corner_of_slope(
        self,
        inner_mask: NDArray[tuple[bool, bool]],
        input_depth_map_with_bump: NDArray[tuple[int, int]],
    ) -> None:
        """Test if the corner of the slope is an extension of x, y"""
        # Arrange
        corner = (
            self.BUMP_CENTER - self.BUMP_SIZE // 2,
            self.BUMP_CENTER - self.BUMP_SIZE // 2,
        )
        expected_corner_value = 1 / np.sqrt(
            (self.BUMP_HEIGHT // 2) ** 2 + (self.BUMP_HEIGHT // 2) ** 2 + 1
        )

        # Act
        surface_normals = compute_surface_normals(input_depth_map_with_bump, 1, 1)

        # Assert

        assert_allclose(
            surface_normals[corner[0], corner[1], 2],
            expected_corner_value,
            atol=self.TOLERANCE,
            err_msg="corner of x and y should have unit normal of x and y",
        )


class TestCalculateLighting:
    TEST_IMAGE_WIDTH = 10
    TEST_IMAGE_HEIGHT = 10
    TOLERANCE = 1e-5

    @pytest.fixture(scope="class")
    def light_vector(self):
        return LightAngle(azimuth=45, elevation=180).vector

    @pytest.fixture(scope="class")
    def observer_vector(self):
        return LightAngle(azimuth=0, elevation=90).vector

    @pytest.fixture(scope="class")
    def base_images(self) -> IMAGE_3_LAYER_STACK_ARRAY:
        nx = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.7)
        ny = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.6)
        nz = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.2)
        return np.stack([nx, ny, nz], axis=-1)

    def test_shape(
        self,
        base_images: IMAGE_3_LAYER_STACK_ARRAY,
        observer_vector: NORMAL_VECTOR,
        light_vector: NORMAL_VECTOR,
    ):
        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert out.shape == (self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT)

    def test_value_range(
        self,
        base_images: IMAGE_3_LAYER_STACK_ARRAY,
        observer_vector: NORMAL_VECTOR,
        light_vector: NORMAL_VECTOR,
    ):
        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert np.all(out >= 0)
        assert np.all(out <= 1)

    def test_constant_normals_give_constant_output(
        self,
        base_images: IMAGE_3_LAYER_STACK_ARRAY,
        observer_vector: NORMAL_VECTOR,
        light_vector: NORMAL_VECTOR,
    ):
        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert np.allclose(out, out[0, 0])

    def test_bump_changes_values(
        self, observer_vector: NORMAL_VECTOR, light_vector: NORMAL_VECTOR
    ):
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
        self, light_source: NORMAL_VECTOR, nx, ny, nz, observer_vector: NORMAL_VECTOR
    ):
        """Opposite direction → diffuse should be 0."""
        # Arrange
        base_images = np.stack([nx, ny, nz], axis=-1)
        # Act
        out = calculate_lighting(light_source, observer_vector, base_images)

        # Assert
        assert np.all(out == 0), "values should be 0."

    def test_specular_maximum_case(self, observer_vector: NORMAL_VECTOR):
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
        self, base_images, observer_vector: NORMAL_VECTOR, light_vector: NORMAL_VECTOR
    ):
        expected_constant = 0.03535534

        # Act
        out = calculate_lighting(light_vector, observer_vector, base_images)

        # Assert
        assert np.allclose(out, expected_constant, atol=self.TOLERANCE)


class TestNormalizeIntensityMap:
    TEST_IMAGE_WIDTH = 10
    TEST_IMAGE_HEIGHT = 12
    TOLERANCE = 1e-5

    @pytest.mark.parametrize(
        "start_value, slope",
        [
            pytest.param(10, 100.0, id="test bigger numbers are reduced"),
            pytest.param(-200, 10.0, id="test negative numbers are upped"),
            pytest.param(100, 0.01, id="small slope is streched over the range"),
        ],
    )
    def test_bigger_numbers(self, start_value: int, slope: float):
        # Arrange
        row = start_value + slope * np.arange(self.TEST_IMAGE_WIDTH)
        image = np.tile(row, (self.TEST_IMAGE_HEIGHT, 1))
        max_val = 255
        min_val = 20
        # Act
        normalized_image = normalize_intensity_map(
            image, max_val=max_val, scale_min=min_val
        )

        # Assert
        assert normalized_image.max() <= max_val
        assert normalized_image.min() >= min_val
        assert normalized_image[0, 0] == normalized_image.min()
        assert normalized_image[9, 9] == normalized_image.max()

    def test_already_normalized_image(self):
        # Arrange
        max_value = 255
        min_val = 20
        image = np.linspace(
            min_val, max_value, num=self.TEST_IMAGE_WIDTH * self.TEST_IMAGE_HEIGHT
        ).reshape(self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT)

        # Act
        normalized = normalize_intensity_map(
            image, max_val=max_value, scale_min=min_val
        )

        # Assert
        assert np.all(normalized >= min_val)
        assert np.all(normalized <= max_value)
        assert np.allclose(image, normalized, atol=self.TOLERANCE), (
            "should be the same output as the already normalized input"
        )


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
    ):
        """A dumb calculator that returns a constant equal to the x-component of the light."""
        return np.sum(surface_normals * light_vector, axis=-1)

    def test_apply_multiple_lights(self):
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
