"""Tests for conversion.surface_comparison.cell_registration.utils."""

import numpy as np
import pytest

from conversion.surface_comparison.cell_registration.utils import (
    convert_grid_cell_to_cell,
    pad_image_array,
    rotated_crop,
    rotate_image,
    canvas_to_image,
    rotated_shape,
)
from conversion.surface_comparison.models import GridCell
from .helpers import make_surface

PIXEL_SIZE = 1e-6
IMAGE_HEIGHT = 850
IMAGE_WIDTH = 600
PAD_HEIGHT = 25
PAD_WIDTH = 25
ANGLES = [-30.0, -7.25, -0.5, 0.0, 0.5, 3.0, 13.0, 45.0]


class TestRotatedShape:
    SHAPE = (211, 173)

    def test_zero_angle_leaves_shape_unchanged(self):
        assert rotated_shape(*self.SHAPE, 0.0) == self.SHAPE

    def test_half_turn_leaves_shape_unchanged(self):
        assert rotated_shape(*self.SHAPE, 180.0) == self.SHAPE

    def test_quarter_turn_swaps_axes(self):
        height, width = self.SHAPE
        assert rotated_shape(height, width, 90.0) == (width, height)

    @pytest.mark.parametrize("angle", [7.25, 30.0, 61.0])
    def test_is_symmetric_in_sign(self, angle):
        assert rotated_shape(*self.SHAPE, angle) == rotated_shape(*self.SHAPE, -angle)

    @pytest.mark.parametrize("angle", ANGLES)
    def test_matches_the_corner_bounding_box(self, angle):
        # Arrange: the canvas must contain the rotated corners of the original image.
        height, width = self.SHAPE
        corners = np.array(
            [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)],
            dtype=float,
        )
        radians = np.radians(angle)
        rotation = np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
        )
        rotated = corners @ rotation.T

        # Act
        rotated_height, rotated_width = rotated_shape(height, width, angle)

        # Assert
        assert rotated_width == round(np.ptp(rotated[:, 0])) + 1
        assert rotated_height == round(np.ptp(rotated[:, 1])) + 1

    @pytest.mark.parametrize("angle", ANGLES)
    def test_predicts_what_rotate_image_produces(self, angle):
        # Arrange
        image = np.zeros(self.SHAPE)

        # Act
        rotated = rotate_image(image, angle, fill_value=0.0)

        # Assert
        assert rotated.shape == rotated_shape(*self.SHAPE, angle)


class TestRotateImage:
    SHAPE = (121, 97)

    @staticmethod
    def _blob(shape, x, y, sigma=4.0):
        """A smooth peak, which survives nearest-neighbour resampling unlike a single pixel."""
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        return np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2))

    @staticmethod
    def _centroid(array):
        """Sub-pixel location of the brightest feature."""
        weights = np.clip(
            np.nan_to_num(array, nan=0.0) - np.nanmax(array) * 0.5, 0, None
        )
        total = weights.sum()
        return (
            float((weights * np.arange(array.shape[1])[None, :]).sum() / total),
            float((weights * np.arange(array.shape[0])[:, None]).sum() / total),
        )

    def test_zero_angle_preserves_the_image(self):
        # Arrange
        image = self._blob(self.SHAPE, 30, 40)

        # Act
        rotated = rotate_image(image, 0.0, fill_value=0.0)

        # Assert
        assert rotated == pytest.approx(image.astype(np.float32))

    def test_returns_float32(self):
        assert rotate_image(np.zeros(self.SHAPE), 5.0).dtype == np.float32

    def test_fills_outside_the_source_rectangle(self):
        # Arrange: the corners of a rotated canvas cannot come from the source image.
        image = np.ones(self.SHAPE)

        # Act
        rotated = rotate_image(image, 30.0, fill_value=-7.0)

        # Assert
        assert rotated[0, 0] == pytest.approx(-7.0)
        assert rotated[-1, -1] == pytest.approx(-7.0)

    def test_fills_with_nan_by_default(self):
        assert np.isnan(rotate_image(np.ones(self.SHAPE), 30.0)[0, 0])

    @pytest.mark.parametrize("angle", ANGLES)
    def test_maps_the_image_centre_onto_the_canvas_centre(self, angle):
        # Arrange
        height, width = self.SHAPE
        image = self._blob(self.SHAPE, (width - 1) / 2, (height - 1) / 2)

        # Act
        rotated = rotate_image(image, angle, fill_value=0.0)

        # Assert
        rotated_height, rotated_width = rotated.shape
        x, y = self._centroid(rotated)
        assert x == pytest.approx((rotated_width - 1) / 2, abs=0.5)
        assert y == pytest.approx((rotated_height - 1) / 2, abs=0.5)

    @pytest.mark.parametrize("angle", ANGLES)
    @pytest.mark.parametrize("position", [(20, 25), (70, 90), (48, 60)])
    def test_canvas_to_image_inverts_the_actual_rotation(self, angle, position):
        """
        The geometry helpers must agree with what ``rotate_image`` really does, not merely with
        each other. Plant a feature, rotate, find it, and map it back.
        """
        # Arrange
        x0, y0 = position
        image = self._blob(self.SHAPE, x0, y0)

        # Act
        rotated = rotate_image(image, angle, fill_value=0.0)
        found_x, found_y = self._centroid(rotated)
        # A one-pixel "cell" centred on the feature: its centre is the point we want back.
        recovered = canvas_to_image(
            found_x - 0.5, found_y - 0.5, (1, 1), self.SHAPE, angle
        )

        # Assert: bounded by nearest-neighbour resampling, not by the mapping.
        assert recovered[0] == pytest.approx(x0, abs=0.5)
        assert recovered[1] == pytest.approx(y0, abs=0.5)


class TestRotatedCrop:
    SHAPE = (140, 160)
    CROP = (48, 52)  # (height, width)

    @pytest.fixture
    def image(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=self.SHAPE).astype(np.float32)

    def test_has_the_requested_shape(self, image):
        # Act
        crop = rotated_crop(image, 12.0, 10, 20, self.CROP[1], self.CROP[0])

        # Assert
        assert crop.shape == self.CROP

    @pytest.mark.parametrize("angle", ANGLES)
    @pytest.mark.parametrize("origin", [(0, 0), (30, 25), (60, 70)])
    def test_matches_the_same_slice_of_the_full_rotation(self, image, angle, origin):
        """
        Cropping must be interchangeable with rotating everything and slicing, since that is what
        makes skipping the full rotation safe.

        Agreement is not bit-exact. The two calls build slightly different affine matrices, so a
        source coordinate landing on a half-pixel boundary can round either way under
        nearest-neighbour sampling. Measured at under 1% of pixels across the angles here, which
        moves a correlation score by far less than the differences already tolerated elsewhere.
        """
        # Arrange
        left, top = origin
        full = rotate_image(image, angle, fill_value=0.0)

        # Act
        crop = rotated_crop(
            image, angle, left, top, self.CROP[1], self.CROP[0], fill_value=0.0
        )

        # Assert
        expected = full[top : top + self.CROP[0], left : left + self.CROP[1]]
        actual = crop[: expected.shape[0], : expected.shape[1]]
        assert np.mean(~np.isclose(actual, expected)) < 0.02

    def test_region_beyond_the_canvas_is_filled(self, image):
        # Arrange: a crop taken entirely off the top-left of the
        # Arrange: a crop taken entirely off the top-left of the rotated canvas.
        # Act
        crop = rotated_crop(
            image, 10.0, -400, -400, self.CROP[1], self.CROP[0], fill_value=-3.0
        )

        # Assert
        assert crop == pytest.approx(np.full(self.CROP, -3.0, dtype=np.float32))

    def test_negative_origin_still_aligns_with_the_full_rotation(self, image):
        # Arrange: a crop straddling the canvas edge must line up where they overlap.
        full = rotate_image(image, 8.0, fill_value=0.0)
        left, top = -10, -12

        # Act
        crop = rotated_crop(
            image, 8.0, left, top, self.CROP[1], self.CROP[0], fill_value=0.0
        )

        # Assert
        overlap = full[0 : self.CROP[0] + top, 0 : self.CROP[1] + left]
        assert np.mean(~np.isclose(crop[-top:, -left:], overlap)) < 0.02


class TestPadImageArray:
    def test_pad_image_array_output_shape(self):
        # Arrange
        array = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        padded = pad_image_array(array, pad_width=PAD_WIDTH, pad_height=PAD_HEIGHT)

        # Assert
        assert padded.shape == (
            IMAGE_HEIGHT + 2 * PAD_HEIGHT,
            IMAGE_WIDTH + 2 * PAD_WIDTH,
        )

    def test_pad_image_array_border_is_nan(self):
        # Arrange
        array = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        padded = pad_image_array(array, pad_width=PAD_WIDTH, pad_height=PAD_HEIGHT)

        # Assert — all border rows/cols should be NaN (default fill)
        assert np.all(np.isnan(padded[:PAD_HEIGHT, :]))
        assert np.all(np.isnan(padded[-PAD_HEIGHT:, :]))
        assert np.all(np.isnan(padded[:, :PAD_WIDTH]))
        assert np.all(np.isnan(padded[:, -PAD_WIDTH:]))

    def test_pad_image_array_interior_matches_original(self):
        # Arrange
        array = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        padded = pad_image_array(array, pad_width=PAD_WIDTH, pad_height=PAD_HEIGHT)

        # Assert
        interior = padded[
            PAD_HEIGHT : PAD_HEIGHT + IMAGE_HEIGHT, PAD_WIDTH : PAD_WIDTH + IMAGE_WIDTH
        ]
        np.testing.assert_array_equal(interior, array)


class TestConvertGridCellToCell:
    def test_convert_grid_cell_to_cell_centers_in_meters(
        self, fully_valid_grid_cell: GridCell
    ):
        # Arrange
        cell = fully_valid_grid_cell

        # Act
        result = convert_grid_cell_to_cell(grid_cell=cell, pixel_size=PIXEL_SIZE)

        # Assert — reference center must equal (top_left + half_cell) * pixel_size
        expected_cx = (cell.top_left[0] + cell.width / 2) * PIXEL_SIZE
        expected_cy = (cell.top_left[1] + cell.height / 2) * PIXEL_SIZE
        assert result.center_reference == pytest.approx((expected_cx, expected_cy))

    def test_convert_grid_cell_to_cell_score_propagated(
        self, fully_valid_grid_cell: GridCell
    ):
        # Arrange
        cell = fully_valid_grid_cell

        # Act
        result = convert_grid_cell_to_cell(grid_cell=cell, pixel_size=PIXEL_SIZE)

        # Assert
        assert result.best_score == pytest.approx(cell.grid_search_params.score)

    def test_convert_grid_cell_to_cell_fill_fraction_propagated(
        self,
        fully_valid_grid_cell: GridCell,
    ):
        # Arrange
        cell = fully_valid_grid_cell

        # Act
        result = convert_grid_cell_to_cell(grid_cell=cell, pixel_size=PIXEL_SIZE)

        # Assert
        assert result.fill_fraction_reference == pytest.approx(cell.fill_fraction)
