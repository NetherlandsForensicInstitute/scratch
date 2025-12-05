import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from conversion.mask import (
    mask_2d_array,
    crop_to_mask,
    _determine_bounding_box,
    mask_and_crop_2d_array,
)
from parsers import ScanImage
from utils.array_definitions import MaskArray
from ..constants import BASELINE_IMAGES_DIR


class TestMask2dArray:
    def test_masks_background_pixels(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.array([[1, 0], [0, 1]])

        result = mask_2d_array(image, mask)

        assert result[0, 0] == 1
        assert result[1, 1] == 4
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 0])

    def test_does_not_mutate_input(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.array([[1, 0], [0, 1]])
        original = image.copy()

        mask_2d_array(image, mask)

        np.testing.assert_array_equal(image, original)

    def test_raises_on_shape_mismatch(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.array([[1, 0, 0], [0, 1, 0]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            mask_2d_array(image, mask)

    def test_full_mask_preserves_all_values(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.ones((2, 2)).astype(bool)

        result = mask_2d_array(image, mask)

        np.testing.assert_array_equal(result, image)

    def test_empty_mask_sets_all_to_nan(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.zeros((2, 2)).astype(bool)

        result = mask_2d_array(image, mask)

        assert np.all(np.isnan(result))


class TestCropToMask:
    def test_crops_to_bounding_box(self):
        image = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
        mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        )

        result = crop_to_mask(image, mask)

        expected = np.array([[1, 2], [3, 4]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_single_pixel_mask(self):
        image = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=float,
        )
        mask = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )

        result = crop_to_mask(image, mask)

        expected = np.array([[5]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_full_mask_returns_full_image(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.ones((2, 2)).astype(bool)

        result = crop_to_mask(image, mask)

        np.testing.assert_array_equal(result, image)


class TestDetermineBoundingBox:
    def test_returns_correct_slices(self):
        mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        )

        x_slice, y_slice = _determine_bounding_box(mask)

        assert y_slice == slice(1, 3)
        assert x_slice == slice(1, 3)

    def test_asymmetric_mask(self):
        mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        x_slice, y_slice = _determine_bounding_box(mask)

        assert y_slice == slice(1, 2)
        assert x_slice == slice(2, 4)

    def test_raises_on_empty_mask(self):
        mask = np.zeros((3, 3))

        with pytest.raises(ValueError, match="Mask is empty"):
            _determine_bounding_box(mask)  # type: ignore


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage, mask_with_nans: MaskArray
):
    verified = np.load(BASELINE_IMAGES_DIR / "masked_cropped_array.npy")
    masked_cropped_image = mask_and_crop_2d_array(
        scan_image_with_nans, mask_with_nans, crop=True
    )
    assert_array_almost_equal(masked_cropped_image.data, verified)
