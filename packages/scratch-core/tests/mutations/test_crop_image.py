from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from mutations.spatial import Crop
import pytest
import numpy as np


class TestMaskBoundingBox:
    """Test the mask_bounding_box property.

    Note: mask_bounding_box returns (x_slice, y_slice), following the convention
    from conversion.mask._determine_bounding_box.
    """

    @pytest.mark.parametrize(
        "mask,expected",
        [
            pytest.param(
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    dtype=bool,
                ),
                (slice(0, 3), slice(0, 3)),
                id="all-ones",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1, 1],
                        [0, 0, 0],
                        [1, 1, 1],
                    ],
                    dtype=bool,
                ),
                (slice(0, 3), slice(0, 3)),
                id="middle-row-false",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ],
                    dtype=bool,
                ),
                (slice(0, 3), slice(0, 3)),
                id="ring-true",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                    ],
                    dtype=bool,
                ),
                (slice(0, 3), slice(0, 3)),
                id="middle-column-false",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=bool,
                ),
                (slice(0, 1), slice(0, 1)),
                id="top-left-corner-only",
            ),
            pytest.param(
                np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                    ],
                    dtype=bool,
                ),
                (slice(2, 3), slice(2, 3)),
                id="bottom-right-corner-only",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1, 1],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=bool,
                ),
                (slice(0, 3), slice(0, 1)),
                id="top-row-only",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    dtype=bool,
                ),
                (slice(0, 1), slice(0, 3)),
                id="left-column-only",
            ),
            pytest.param(
                np.array(
                    [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                    ],
                    dtype=bool,
                ),
                (slice(1, 2), slice(1, 2)),
                id="center-pixel-only",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                        [0, 0, 0],
                    ],
                    dtype=bool,
                ),
                (slice(0, 2), slice(0, 2)),
                id="top-left-2x2",
            ),
            pytest.param(
                np.array(
                    [
                        [0, 0, 0],
                        [0, 1, 1],
                        [0, 1, 1],
                    ],
                    dtype=bool,
                ),
                (slice(1, 3), slice(1, 3)),
                id="bottom-right-2x2",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    dtype=bool,
                ),
                (slice(0, 3), slice(0, 3)),
                id="L-shape",
            ),
            pytest.param(
                np.array(
                    [
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                    ],
                    dtype=bool,
                ),
                (slice(1, 2), slice(0, 3)),
                id="middle-column-only",
            ),
            pytest.param(
                np.array(
                    [
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                    ],
                    dtype=bool,
                ),
                (slice(0, 3), slice(0, 3)),
                id="scattered-diagonal-corners",
            ),
            pytest.param(
                np.array(
                    [
                        [0, 0, 1],
                        [0, 0, 0],
                        [1, 0, 0],
                    ],
                    dtype=bool,
                ),
                (slice(0, 3), slice(0, 3)),
                id="scattered-anti-diagonal-corners",
            ),
        ],
    )
    def test_mask_bounding_box_subset(
        self, mask: BinaryMask, expected: tuple[slice, slice], scan_image_factory
    ) -> None:
        # Arrange
        scan_image = scan_image_factory(mask=mask)
        # Act
        slices = scan_image.mask_bounding_box
        # assert
        assert slices == expected


class TestCropImage:
    @pytest.fixture
    def scan_image(self) -> ScanImage:
        data = np.arange(16, dtype=np.float32).reshape((4, 4))
        return ScanImage(data=data.copy(), scale_x=1, scale_y=1)

    def test_crop_applies_bounding_box(self, scan_image: ScanImage):
        # Arrange
        mask = np.ones(scan_image.data.shape, dtype=bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
        crop = Crop(crop=mask)

        # Act
        result = crop(scan_image).unwrap()

        # Assert
        removed_border = np.array([[5, 6], [9, 10]], dtype=scan_image.data.dtype)
        assert result.data.shape == (2, 2)
        np.testing.assert_array_equal(result.data, removed_border)
        assert result.scale_x == scan_image.scale_x, (
            "scale should be the same (unchanged)"
        )
        assert result.scale_y == scan_image.scale_y, (
            "scale should be the same (unchanged)"
        )

    def test_crop_skipped_when_predicate_true(self, scan_image: ScanImage):
        # Arrange
        crop = Crop(crop=(np.zeros(scan_image.data.shape, dtype=np.bool)))
        # Act
        result = crop(scan_image).unwrap()
        # Assert
        assert result is scan_image
        np.testing.assert_array_equal(result.data, scan_image.data)
