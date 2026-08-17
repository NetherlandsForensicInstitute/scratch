"""Tests for conversion.surface_comparison.cell_registration.geometry."""

import numpy as np
import pytest

from conversion.resample import resample_array_2d_nan_aware
from conversion.surface_comparison.cell_registration.geometry import (
    compute_rotated_shape,
    crop_rotated_image,
    map_canvas_to_image,
    map_coarse_to_full,
    map_image_to_canvas,
    pad_image_array,
    rotate_image,
)

from .helpers import make_surface

IMAGE_HEIGHT = 850
IMAGE_WIDTH = 600
PAD_HEIGHT = 25
PAD_WIDTH = 25
ANGLES = [-30.0, -7.25, -0.5, 0.0, 0.5, 3.0, 13.0, 45.0]


class TestComputeRotatedShape:
    SHAPE = (211, 173)

    def test_zero_angle_leaves_shape_unchanged(self):
        assert compute_rotated_shape(*self.SHAPE, 0.0) == self.SHAPE

    def test_half_turn_leaves_shape_unchanged(self):
        assert compute_rotated_shape(*self.SHAPE, 180.0) == self.SHAPE

    def test_quarter_turn_swaps_axes(self):
        height, width = self.SHAPE
        assert compute_rotated_shape(height, width, 90.0) == (width, height)

    @pytest.mark.parametrize("angle", [7.25, 30.0, 61.0])
    def test_is_symmetric_in_sign(self, angle):
        assert compute_rotated_shape(*self.SHAPE, angle) == compute_rotated_shape(
            *self.SHAPE, -angle
        )

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
        rotated_height, rotated_width = compute_rotated_shape(height, width, angle)

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
        assert rotated.shape == compute_rotated_shape(*self.SHAPE, angle)


class TestRotateImage:
    SHAPE = (121, 97)

    @staticmethod
    def _blob(shape, x, y, sigma=4.0):
        """A smooth peak, which survives nearest-neighbor resampling unlike a single pixel."""
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
    def test_maps_the_image_center_onto_the_canvas_center(self, angle):
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
    def test_map_canvas_to_image_inverts_the_actual_rotation(self, angle, position):
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
        # A one-pixel "cell" centered on the feature: its center is the point we want back.
        recovered = map_canvas_to_image(
            found_x - 0.5, found_y - 0.5, (1, 1), self.SHAPE, angle
        )

        # Assert: bounded by nearest-neighbor resampling, not by the mapping.
        assert recovered[0] == pytest.approx(x0, abs=0.5)
        assert recovered[1] == pytest.approx(y0, abs=0.5)


class TestCropRotatedImage:
    SHAPE = (140, 160)
    CROP = (48, 52)  # (height, width)

    @pytest.fixture
    def image(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=self.SHAPE).astype(np.float32)

    def test_has_the_requested_shape(self, image):
        # Act
        crop = crop_rotated_image(image, 12.0, 10, 20, self.CROP[1], self.CROP[0])

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
        nearest-neighbor sampling. Measured at under 1% of pixels across the angles here, which
        moves a correlation score by far less than the differences already tolerated elsewhere.
        """
        # Arrange
        left, top = origin
        full = rotate_image(image, angle, fill_value=0.0)

        # Act
        crop = crop_rotated_image(
            image, angle, left, top, self.CROP[1], self.CROP[0], fill_value=0.0
        )

        # Assert
        expected = full[top : top + self.CROP[0], left : left + self.CROP[1]]
        actual = crop[: expected.shape[0], : expected.shape[1]]
        assert np.mean(~np.isclose(actual, expected)) < 0.02

    def test_region_beyond_the_canvas_is_filled(self, image):
        # Arrange / Act: a crop taken entirely off the top-left of the rotated canvas.
        crop = crop_rotated_image(
            image, 10.0, -400, -400, self.CROP[1], self.CROP[0], fill_value=-3.0
        )

        # Assert
        assert crop == pytest.approx(np.full(self.CROP, -3.0, dtype=np.float32))

    def test_negative_origin_still_aligns_with_the_full_rotation(self, image):
        # Arrange: a crop straddling the canvas edge must line up where they overlap.
        full = rotate_image(image, 8.0, fill_value=0.0)
        left, top = -10, -12

        # Act
        crop = crop_rotated_image(
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


class TestMapCoarseToFull:
    """
    One coarse pixel covers a ``cap_factor``-wide block of full-resolution pixels, and maps onto
    the *center* of that block — the convention cv2.resize resamples on.
    """

    @staticmethod
    def _block_center(coarse_index: int, cap_factor: int) -> float:
        """Center of the full-resolution block that unpadded coarse pixel *coarse_index* covers."""
        return float(
            np.arange(coarse_index * cap_factor, (coarse_index + 1) * cap_factor).mean()
        )

    def test_factor_one_and_equal_padding_is_identity(self):
        assert map_coarse_to_full(7.0, 1, 3, 3) == pytest.approx(7.0)

    @pytest.mark.parametrize(
        ("coarse_index", "cap_factor"), [(0, 4), (1, 4), (3, 4), (3, 2)]
    )
    def test_scales_the_offset_from_the_padded_origin(self, coarse_index, cap_factor):
        # Arrange: two coarse pixels of padding against ten full ones.
        coarse_padding, full_padding = 2, 10

        # Act
        result = map_coarse_to_full(
            coarse_padding + coarse_index, cap_factor, coarse_padding, full_padding
        )

        # Assert
        assert result == pytest.approx(
            full_padding + self._block_center(coarse_index, cap_factor)
        )

    @pytest.mark.parametrize("cap_factor", [1, 2, 4, 6])
    def test_the_first_image_pixel_maps_onto_its_block_center(self, cap_factor):
        # Arrange: the image starts at the padding on both canvases.
        coarse_padding, full_padding = 9, 50

        # Act
        result = map_coarse_to_full(
            float(coarse_padding), cap_factor, coarse_padding, full_padding
        )

        # Assert: the block's center, not its left edge.
        assert result == pytest.approx(full_padding + self._block_center(0, cap_factor))

    def test_removes_the_coarse_padding_before_scaling_it(self):
        # Arrange: 9 coarse pixels of padding cover 54 full pixels, not the 50 of the full canvas,
        # so scaling the padding along with the coordinate would drift by 4 pixels.
        factor, coarse_padding, full_padding = 6.0, 9, 50

        # Act: the first image pixel, and one coarse pixel further in.
        first = map_coarse_to_full(
            float(coarse_padding), factor, coarse_padding, full_padding
        )
        second = map_coarse_to_full(
            coarse_padding + 1.0, factor, coarse_padding, full_padding
        )

        # Assert
        assert first == pytest.approx(full_padding + self._block_center(0, 6))
        assert second - first == pytest.approx(factor)

    @pytest.mark.parametrize("cap_factor", [2, 4, 8])
    def test_agrees_with_the_grid_the_coarse_image_is_resampled_on(self, cap_factor):
        """
        The mapping is only correct relative to how the coarse canvas was built. Area-averaging a
        ramp whose values are each pixel's own x coordinate leaves every coarse pixel holding the
        full-resolution coordinate at its center — exactly what the mapping must return.
        """
        # Arrange
        width = 32 * cap_factor
        ramp = np.tile(np.arange(width, dtype=float), (4, 1))

        # Act: downsample exactly as build_coarse_stage does.
        coarse = resample_array_2d_nan_aware(ramp, (cap_factor, cap_factor), "area")

        # Assert
        for coarse_x in range(coarse.shape[1]):
            assert map_coarse_to_full(coarse_x, cap_factor, 0, 0) == pytest.approx(
                coarse[0, coarse_x], abs=1e-3
            )


class TestCoordinateMapping:
    PADDED_SHAPE = (601, 517)
    CELL_SHAPE = (80, 90)
    POINT_IN_PADDED = (137.0, 288.0)
    CENTER_IN_IMAGE = (250.0, 300.0)

    def test_angle_zero_is_identity(self):
        # Act
        x, y = map_canvas_to_image(
            *self.POINT_IN_PADDED,
            cell_shape=self.CELL_SHAPE,
            image_shape=self.PADDED_SHAPE,
            angle_deg=0.0,
        )

        # Assert
        assert x == pytest.approx(self.POINT_IN_PADDED[0] + self.CELL_SHAPE[1] / 2)
        assert y == pytest.approx(self.POINT_IN_PADDED[1] + self.CELL_SHAPE[0] / 2)

    @pytest.mark.parametrize("angle_deg", [-15.0, -3.5, 0.0, 0.5, 7.25, 20.0])
    def test_round_trip(self, angle_deg):
        left, top = map_image_to_canvas(
            *self.CENTER_IN_IMAGE,
            cell_shape=self.CELL_SHAPE,
            image_shape=self.PADDED_SHAPE,
            angle_deg=angle_deg,
        )
        x, y = map_canvas_to_image(
            left, top, self.CELL_SHAPE, self.PADDED_SHAPE, angle_deg
        )
        assert x == pytest.approx(self.CENTER_IN_IMAGE[0])
        assert y == pytest.approx(self.CENTER_IN_IMAGE[1])

    @pytest.mark.parametrize("angle", [30.0, -60.0, 90.0])
    def test_round_trip_recovers_original_point(self, angle: float):
        """
        Analytically forward-map a known point onto the rotated canvas, then verify
        map_canvas_to_image recovers it.
        """
        # Arrange
        height, width = self.PADDED_SHAPE
        rotated_height, rotated_width = compute_rotated_shape(height, width, angle)
        cx_pad, cy_pad = (width - 1) / 2, (height - 1) / 2
        cx_rot, cy_rot = (rotated_width - 1) / 2, (rotated_height - 1) / 2
        px, py = self.POINT_IN_PADDED
        a = np.radians(angle)
        fwd_x = np.cos(a) * (px - cx_pad) - np.sin(a) * (py - cy_pad) + cx_rot
        fwd_y = np.sin(a) * (px - cx_pad) + np.cos(a) * (py - cy_pad) + cy_rot

        # Act
        recovered_x, recovered_y = map_canvas_to_image(
            fwd_x,
            fwd_y,
            cell_shape=(0, 0),
            image_shape=self.PADDED_SHAPE,
            angle_deg=angle,
        )

        # Assert
        assert recovered_x == pytest.approx(px, abs=1e-6)
        assert recovered_y == pytest.approx(py, abs=1e-6)
