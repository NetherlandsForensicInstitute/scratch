from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from conversion.mask import (
    mask_2d_array,
    crop_to_mask,
    _determine_bounding_box,
    mask_and_crop_2d_array,
    mask_and_crop_scan_image,
)
from container_models.scan_image import ScanImage
from container_models.base import MaskArray


class TestMask2dArray:
    def test_mask_sets_background_pixels_to_nan(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.array([[1, 0], [0, 1]], dtype=bool)

        result = mask_2d_array(image, mask)

        np.testing.assert_array_equal(result, np.array([[1, np.nan], [np.nan, 4]]))

    def test_does_not_mutate_input(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.array([[1, 0], [0, 1]], dtype=bool)
        original = image.copy()

        _ = mask_2d_array(image, mask)

        np.testing.assert_array_equal(image, original)

    def test_raises_on_shape_mismatch(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            mask_2d_array(image, mask)

    def test_full_mask_preserves_all_values(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.ones((2, 2), dtype=bool)

        result = mask_2d_array(image, mask)

        np.testing.assert_array_equal(result, image)

    def test_empty_mask_sets_all_to_nan(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.zeros((2, 2), dtype=bool)

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
            ],
            dtype=bool,
        )

        result = crop_to_mask(image, mask)

        expected = np.array([[1, 2], [3, 4]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_single_pixel_mask_crops_to_single_pixel_output(self):
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
            ],
            dtype=bool,
        )

        result = crop_to_mask(image, mask)

        expected = np.array([[5]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_full_mask_returns_full_image(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.ones((2, 2), dtype=bool)

        result = crop_to_mask(image, mask)

        np.testing.assert_array_equal(result, image)


class TestDetermineBoundingBox:
    @pytest.fixture
    def mask(self) -> MaskArray:
        mask = np.zeros((10, 10), dtype=float)
        mask[2:8, 2:8] = True
        return mask

    @pytest.fixture
    def asymmetric_mask(self) -> MaskArray:
        mask = np.zeros((10, 20), dtype=float)
        mask[2:8, 3:15] = True
        return mask

    @pytest.mark.parametrize(
        "margin, output_slice",
        [
            pytest.param(None, slice(6, 6), id="Normal bounding box, no margin"),
            pytest.param(0, slice(6, 6), id="Normal bounding box, no marginx"),
            pytest.param(1, slice(4, 4), id="Decrease bounding box"),
            pytest.param(-1, slice(8, 8), id="Increase bounding box"),
            pytest.param(-5, slice(10, 10), id="Margin bounding box out of bounds"),
        ],
    )
    def test_bounding_box_slices_with_margin(
        self, mask: MaskArray, margin: int, output_slice: slice
    ):
        """Test bounding boxes with different margins."""
        x_slice, y_slice = _determine_bounding_box(mask, margin)

        assert y_slice == output_slice
        assert x_slice == output_slice

    @pytest.mark.parametrize(
        "margin, output_slice_y, output_slice_x",
        [
            pytest.param(
                None, slice(2, 8), slice(3, 15), id="Normal bounding box, no margin"
            ),
            pytest.param(
                0, slice(2, 8), slice(3, 15), id="Normal bounding box, no margin"
            ),
            pytest.param(1, slice(3, 7), slice(4, 14), id="Decrease bounding box"),
            pytest.param(-1, slice(1, 9), slice(2, 16), id="Increase bounding box"),
            pytest.param(
                -20, slice(0, 10), slice(0, 20), id="Margin bounding box out of bounds"
            ),
        ],
    )
    def test_asymmetric_bounding_box_slices_with_margin(
        self,
        asymmetric_mask: MaskArray,
        margin: int,
        output_slice_y: slice,
        output_slice_x: slice,
    ):
        """Test bounding boxes with different margins."""
        x_slice, y_slice = _determine_bounding_box(asymmetric_mask, margin)

        assert y_slice == output_slice_y
        assert x_slice == output_slice_x

    def test_bounding_box_slices_only_foreground(self):
        mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        )

        x_slice, y_slice = _determine_bounding_box(mask)

        assert y_slice == slice(1, 3)
        assert x_slice == slice(1, 3)

    def test_asymmetric_mask_slices_only_foreground(self, asymmetric_mask: MaskArray):
        x_slice, y_slice = _determine_bounding_box(asymmetric_mask)

        assert y_slice == slice(2, 8)
        assert x_slice == slice(3, 15)

    def test_raises_on_empty_mask(self):
        mask = np.zeros((3, 3), dtype=bool)

        with pytest.raises(ValueError, match="Mask is empty"):
            _determine_bounding_box(mask)


class TestCropScanImage:
    @pytest.mark.integration
    def test_mask_scan_image(
        self, scan_image_replica: ScanImage, mask_array: MaskArray
    ):
        masked_scan_image = mask_and_crop_scan_image(
            scan_image=scan_image_replica, mask=mask_array, crop=False
        )
        nans = np.isnan(scan_image_replica.data) | ~mask_array
        assert np.array_equal(nans, np.isnan(masked_scan_image.data))

    @pytest.mark.integration
    def test_crop_scan_image(
        self, scan_image_replica: ScanImage, mask_array: MaskArray
    ):
        cropped_scan_image = mask_and_crop_scan_image(
            scan_image=scan_image_replica, mask=mask_array, crop=True
        )
        assert cropped_scan_image.width < scan_image_replica.width
        assert cropped_scan_image.height < scan_image_replica.height


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage, mask_array: MaskArray, baseline_images_dir: Path
):
    verified = np.load(baseline_images_dir / "masked_cropped_array.npy")
    masked_cropped_array = mask_and_crop_2d_array(
        scan_image_with_nans.data, mask_array, crop=True
    )
    assert_array_almost_equal(masked_cropped_array, verified)
