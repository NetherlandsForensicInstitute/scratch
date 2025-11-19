import numpy as np
import pytest
from numpy._typing import NDArray
from numpy.testing import assert_allclose

from conversion._translation import calculate_surface, getv
from surface_conversion import convert_image_to_slope_map


class TestSurfaceSlopeConversion:
    TEST_IMAGE_WIDTH = 20
    TEST_IMAGE_HEIGHT = 20
    TOLERANCE = 1e-6

    @pytest.fixture(scope="class")
    def inner_mask(self) -> NDArray[tuple[int, int]]:
        inner_mask = np.zeros(
            (self.TEST_IMAGE_HEIGHT, self.TEST_IMAGE_WIDTH), dtype=bool
        )
        inner_mask[1:-1, 1:-1] = True
        return inner_mask

    def test_slope_has_nan_border(self, inner_mask: NDArray[tuple[int, int]]) -> None:
        """The image is 1 pixel smaller on all sides due to the slope calculation.
        This is filled with NaN values to get the same shape as original image"""
        # Arrange
        input_image = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        outer_mask = ~inner_mask

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_image, 1, 1)

        # Assert
        assert not np.any(np.isnan(n1[inner_mask])), (
            "inner row and columns should have a number"
        )
        assert not np.any(np.isnan(n2[inner_mask])), (
            "outer row and columns should have a number"
        )
        assert not np.any(np.isnan(n3[inner_mask])), (
            "outer row and columns should have a number"
        )
        assert np.all(np.isnan(n1[outer_mask])), (
            "all outer row and columns should be NaN"
        )
        assert np.all(np.isnan(n2[outer_mask])), (
            "all outer row and columns should be NaN"
        )
        assert np.all(np.isnan(n3[outer_mask])), (
            "all outer row and columns should be NaN"
        )

    def test_flat_surface_returns_upward_normal(
        self, inner_mask: NDArray[tuple[int, int]]
    ) -> None:
        """Given a flat surface the depth map should also be flat."""
        # Arrange
        input_image = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_image, 1, 1)

        # Assert
        assert n1.shape == input_image.shape
        assert_allclose(n1[inner_mask], 0), "innerside should be 0 (no x direction)"
        assert_allclose(n2[inner_mask], 0), "innerside should be 0 (no y direction)"
        assert_allclose(n3[inner_mask], 1), "innerside should be 1 (no z direction)"

    @pytest.mark.parametrize(
        "step_x, step_y",
        [
            pytest.param(2, 0, id="step increase in x"),
            pytest.param(
                0,
                2,
                id="step increase in y",
            ),
            pytest.param(2, 2, id="step increase in x and y"),
            pytest.param(2, -2, id="positive and negative steps"),
            pytest.param(-2, -2, id="negative x and y steps"),
        ],
    )
    def test_linear_slope(
        self, step_x: int, step_y: int, inner_mask: NDArray[tuple[int, int]]
    ) -> None:
        """Test linear slopes in X, Y, or both directions."""
        # Arrange
        norm = np.sqrt(step_x**2 + step_y**2 + 1)
        expected_n1 = -step_x / norm
        expected_n2 = step_y / norm
        expected_n3 = 1 / norm
        x_vals = np.arange(self.TEST_IMAGE_WIDTH) * step_x
        y_vals = np.arange(self.TEST_IMAGE_HEIGHT) * step_y
        input_image = y_vals[:, None] + x_vals[None, :]

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_image, xdim=1, ydim=1)

        # Assert
        (
            assert_allclose(n1[inner_mask], expected_n1, atol=self.TOLERANCE),
            (f"expected continuous n1 slope of {expected_n1}"),
        )
        (
            assert_allclose(n2[inner_mask], expected_n2, atol=self.TOLERANCE),
            (f"expected continuous n2 slope of {expected_n2}"),
        )
        (
            assert_allclose(n3[inner_mask], expected_n3, atol=self.TOLERANCE),
            (f"expected continuous n3 slope of {expected_n3}"),
        )

    def test_local_slope_location(self, inner_mask: NDArray[tuple[int, int]]) -> None:
        """Check that slope calculation is localized to the bump coordinates."""
        # Arrange
        image_size = self.TEST_IMAGE_WIDTH
        center_row = image_size // 2
        center_col = image_size // 2
        bump_size = 4
        nan_offset = 1
        input_depth_map = np.zeros((image_size, image_size))

        bump_height = 6
        bump_rows = slice(center_row - bump_size // 2, center_row + bump_size // 2)
        bump_cols = slice(center_col - bump_size // 2, center_col + bump_size // 2)
        input_depth_map[bump_rows, bump_cols] = bump_height

        bump_mask = np.zeros_like(input_depth_map, dtype=bool)
        bump_mask[
            center_col - bump_size // 2 - nan_offset : center_col
            + bump_size // 2
            + nan_offset,
            center_row - bump_size // 2 : center_row + bump_size // 2,
        ] = True
        bump_mask[
            center_col - bump_size // 2 : center_col + bump_size // 2,
            center_row - bump_size // 2 - nan_offset : center_row
            + bump_size // 2
            + nan_offset,
        ] = True
        outside_bump_mask = ~bump_mask & inner_mask

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_depth_map, xdim=1, ydim=1)

        # Assert
        assert np.any(np.abs(n1[bump_mask]) > 0), "n1 should have slope inside bump"
        assert np.any(np.abs(n2[bump_mask]) > 0), "n2 should have slope inside bump"
        assert np.any(np.abs(n3[bump_mask]) != 1), (
            "n3 should deviate from 1 inside bump"
        )
        (
            assert_allclose(n1[outside_bump_mask], 0, atol=self.TOLERANCE),
            "outside the bumb X should be 0",
        )
        (
            assert_allclose(n2[outside_bump_mask], 0, atol=self.TOLERANCE),
            "outside the bumb Y should be 0",
        )
        (
            assert_allclose(n3[outside_bump_mask], 1, atol=self.TOLERANCE),
            "outside the bumb Z should be 1",
        )

    def test_corner_of_slope(self, inner_mask: NDArray[tuple[int, int]]):
        """Test if the corner of the slope is an extension of x, y"""
        image_size = self.TEST_IMAGE_WIDTH
        center_row = image_size // 2
        center_col = image_size // 2
        bump_size = 4
        bump_height = 6
        input_depth_map = np.zeros((image_size, image_size))

        expected_corner_value = 1 / np.sqrt(
            (bump_height // 2) ** 2 + (bump_height // 2) ** 2 + 1
        )

        bump_rows = slice(center_row - bump_size // 2, center_row + bump_size // 2)
        bump_cols = slice(center_col - bump_size // 2, center_col + bump_size // 2)
        input_depth_map[bump_rows, bump_cols] = bump_height

        nan_offset = 1
        bump_mask = np.zeros_like(input_depth_map, dtype=bool)
        bump_mask[
            center_col - bump_size // 2 - nan_offset : center_col + bump_size // 2,
            center_row - bump_size // 2 - nan_offset : center_row + bump_size // 2,
        ] = True
        ~bump_mask & inner_mask

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_depth_map, xdim=1, ydim=1)

        # Assert
        corner = (center_row - bump_size // 2, center_col - bump_size // 2)

        (
            assert_allclose(n3[corner], expected_corner_value, atol=self.TOLERANCE),
            "corner of x and y should have unit normal of x and y",
        )


class TestCalculateSurface:
    TEST_IMAGE_WIDTH = 10
    TEST_IMAGE_HEIGHT = 10
    TOLERANCE = 1e-5

    @pytest.fixture(scope="class")
    def base_images(self):
        n1 = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.7)
        n2 = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.6)
        n3 = np.full((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), 0.2)
        return (n1, n2, n3)

    def test_shape(self, base_images):
        # Arrange
        n1, n2, n3 = base_images
        light_source = getv(45, 180)
        observer_vector = getv(0, 90)

        # Act
        out = calculate_surface(light_source, observer_vector, n1, n2, n3)

        # Assert
        assert out.shape == (self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT)

    def test_value_range(self, base_images):
        # Arrange
        n1, n2, n3 = base_images
        light_source = getv(45, 180)
        observer_vector = getv(0, 90)

        # Act
        out = calculate_surface(light_source, observer_vector, n1, n2, n3)

        # Assert
        assert np.all(out >= 0)
        assert np.all(out <= 1)

    def test_constant_normals_give_constant_output(self, base_images):
        # Arrange
        n1, n2, n3 = base_images
        light_source = getv(10, 30)
        observer_vector = getv(0, 90)

        # Act
        out = calculate_surface(light_source, observer_vector, n1, n2, n3)

        # Assert
        assert np.allclose(out, out[0, 0])

    def test_bump_changes_values(self):
        """Test that the shader reacts per pixel by giving a bump in the normals."""
        # Arrange
        n1 = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        n2 = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        n3 = np.ones((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        n3[self.TEST_IMAGE_WIDTH // 2, self.TEST_IMAGE_HEIGHT // 2] = 1.3
        light_source = getv(45, 45)
        observer_vector = getv(0, 90)

        # Act
        out = calculate_surface(light_source, observer_vector, n1, n2, n3)

        # Assert
        center = out[self.TEST_IMAGE_WIDTH // 2, self.TEST_IMAGE_HEIGHT // 2]
        border = out[0, 0]
        assert center != border

    def test_diffuse_clamps_to_zero(self):
        """Opposite direction → diffuse should be 0."""
        # Arrange
        n1 = np.ones((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        n2 = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        n3 = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        light_source = np.array([-1.0, 0.0, 0.0])
        observer_vector = np.array([0.0, 0.0, 1.0])

        # Act
        out = calculate_surface(light_source, observer_vector, n1, n2, n3)

        # Assert
        assert np.all(out == 0)

    def test_specular_maximum_case(self):
        """If light, observer, and normal all align, specular should be maximal."""
        # Arrange
        n1 = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        n2 = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        n3 = np.ones((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        light_source = np.array([0.0, 0.0, 1.0])
        observer_vector = np.array([0.0, 0.0, 1.0])

        # Act
        out = calculate_surface(light_source, observer_vector, n1, n2, n3)

        # Assert
        assert np.allclose(out, 1.0), "(diffuse=1, specular=1), output = (1+1)/2 = 1"

    def test_lighting_known_value(self, base_images):
        # Arrange
        n1, n2, n3 = base_images
        light_source = np.array([1.0, 0.0, 0.0])
        observer_vector = np.array([0.0, 1.0, 0.0])
        expected_constant = 0.46335

        # Act
        out = calculate_surface(light_source, observer_vector, n1, n2, n3)

        # Assert
        assert np.allclose(out, expected_constant, atol=self.TOLERANCE)
