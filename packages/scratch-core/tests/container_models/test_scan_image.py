from functools import partial
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError
import pytest
from scipy.constants import micro

from container_models.scan_image import ScanImage

# TODO: Add tests for ScanImage


DATA: Final[NDArray[np.floating]] = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
)


@pytest.fixture
def scan_image_factory():
    """Factory fixture for ScanImage with scale parameters pre-filled."""
    return partial(ScanImage, data=DATA, scale_x=1 * micro, scale_y=1 * micro)


class TestScanImageConstruction:
    """Test ScanImage creation and validation."""

    def test_create_without_mask(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        scan_image = ScanImage(data=data, scale_x=1 * micro, scale_y=1 * micro)

        assert scan_image.mask is None

    def test_create_with_mask(self):
        # Arrange
        mask = np.ones(DATA.shape, dtype=bool)
        # Act
        scan_image = ScanImage(
            data=DATA, mask=mask, scale_x=1 * micro, scale_y=1 * micro
        )

        assert np.array_equal(scan_image.mask, mask)  # type: ignore
        assert np.array_equal(scan_image.data, DATA)

    def test_mask_and_image_not_equal_in_size(self):
        # Arrange
        mask = np.array([DATA[0]])

        # Act / Assert
        with pytest.raises(ValidationError) as e:
            _ = ScanImage(data=DATA, mask=mask, scale_x=1 * micro, scale_y=1 * micro)
        assert (
            f"The shape of the data {DATA.shape} does not match the shape of the mask {mask.shape}."
            in str(e.value)
        )

    @pytest.mark.parametrize(
        "mask",
        [
            pytest.param(np.ones((2,)), id="1d_mask"),
            pytest.param(np.ones((2, 2, 2), dtype=bool), id="3d_mask"),
        ],
    )
    def test_mask_not_2d_raises_error(self, mask):
        # Act / Assert
        with pytest.raises(
            ValidationError, match="Array shape mismatch, expected 2 dimension"
        ):
            _ = ScanImage(data=DATA, mask=mask, scale_x=1 * micro, scale_y=1 * micro)


class TestApplyMaskImage:
    """Test the apply_mask_image method."""

    @pytest.mark.parametrize(
        "mask,expected",
        [
            pytest.param(
                np.array(
                    [
                        [1, 0, 1],
                        [0, 1, 0],
                        [1, 1, 0],
                    ],
                    dtype=bool,
                ),
                np.array(
                    [
                        [1.0, np.nan, 3.0],
                        [np.nan, 5.0, np.nan],
                        [7.0, 8.0, np.nan],
                    ]
                ),
                id="masked_pixels_set_to_nan",
            ),
            pytest.param(
                np.zeros((3, 3), dtype=bool),
                np.full((3, 3), np.nan),
                id="all_false_mask",
            ),
            pytest.param(
                np.ones((3, 3), dtype=bool),
                DATA,
                id="unchanged_data_with_all_true_mask",
            ),
        ],
    )
    def test_apply_mask_is_masking_scan_image(
        self,
        mask: np.ndarray,
        expected: np.ndarray,
        scan_image_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        scan_image = scan_image_factory(mask=mask)
        # Act
        scan_image.apply_mask_image()
        # Assert
        assert np.array_equal(scan_image.data, expected, equal_nan=True), (
            "All False boolean in mask should result to an np.nan in data"
        )
        assert "Applying mask to scan_image" in caplog.messages

    def test_apply_mask_without_mask_raises_error(self, scan_image_factory):
        # Arrange
        scan_image = scan_image_factory()
        # Act / Assert
        with pytest.raises(ValueError, match="Mask is required"):
            scan_image.apply_mask_image()


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
        self, mask: NDArray, expected: tuple[slice, slice], scan_image_factory
    ) -> None:
        # Arrange
        scan_image = scan_image_factory(mask=mask)
        # Act
        slices = scan_image.mask_bounding_box
        # assert
        # Returns (x_slice for cols, y_slice for rows)
        assert slices == expected

    @pytest.mark.parametrize(
        "mask,error_match",
        [
            pytest.param(None, "Mask is required", id="no-mask"),
            pytest.param(
                np.zeros(DATA.shape, dtype=bool), "zero-size array", id="all-zeros-mask"
            ),
        ],
    )
    def test_mask_bounding_box_raises_error(
        self, mask, error_match, scan_image_factory
    ) -> None:
        scan_image = scan_image_factory(mask=mask)

        with pytest.raises(ValueError, match=error_match):
            _ = scan_image.mask_bounding_box
