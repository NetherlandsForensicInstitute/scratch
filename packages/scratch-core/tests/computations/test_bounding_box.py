from computations.spatial import get_bounding_box
from container_models.base import BinaryMask
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
        self, mask: BinaryMask, expected: tuple[slice, slice]
    ) -> None:
        # Act
        slices = get_bounding_box(mask=mask)
        # assert
        assert slices == expected
