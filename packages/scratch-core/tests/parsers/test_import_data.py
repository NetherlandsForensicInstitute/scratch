from collections.abc import Callable
from pathlib import Path
from typing import Any
from scipy.constants import micro
from scipy.io import savemat
from numpy import array, uint8, array_equal

import pytest

from models.enums import ImageType
from models.image import ImageData
from parsers.import_data import _load_mat_file


def compare_image_data(actual: ImageData, expected: ImageData) -> bool:
    """Compare ImageData objects, handling numpy arrays properly."""

    def assert_array(value, comparator):
        return not (actual.depth_data is None) ^ (
            expected.depth_data is None
        ) and array_equal(value, comparator)

    return (
        # Compare scalar fields
        actual.type == expected.type
        and actual.mark_type == expected.mark_type
        and actual.xdim == expected.xdim
        and actual.ydim == expected.ydim
        # Compare array fields using numpy's array_equal
        and assert_array(actual.depth_data, expected.depth_data)
        and assert_array(actual.texture_data, expected.texture_data)
    )


@pytest.fixture(scope="session")
def create_mat_file(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str, dict[str, Any]], Path]:
    path = tmp_path_factory.mktemp("matfiles", numbered=False)

    def wrapper(name: str, data: dict[str, Any]) -> Path:
        matfile = path / f"{name}.mat"
        savemat(str(matfile), data)
        return matfile

    return wrapper


def test_load_mat_file_empty(
    create_mat_file: Callable[[str, dict[str, Any]], Path],
) -> None:
    """Test that loading an empty MAT file raises ValueError."""
    path = create_mat_file("empty", {})
    with pytest.raises(ValueError, match="MAT file must contain 'type' field"):
        _load_mat_file(path)


@pytest.mark.parametrize(
    "name, data, expected",
    (
        pytest.param(
            "surface",
            {
                "depth_data": array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
                "xdim": micro,
                "ydim": micro,
                "type": "surface",
            },
            ImageData(
                type=ImageType.SURFACE,
                depth_data=array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
                xdim=micro,
                ydim=micro,
            ),
            id="surface",
        ),
        pytest.param(
            "profile",
            {
                "depth_data": array([1.0, 2.0, 3.0, 4.0, 5.0]),
                "xdim": micro,
                "type": "profile",
            },
            ImageData(
                type=ImageType.PROFILE,
                depth_data=array([1.0, 2.0, 3.0, 4.0, 5.0]),
                xdim=micro,
            ),
            id="profile",
        ),
        pytest.param(
            "texture",
            {
                "depth_data": array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
                "texture_data": array(
                    [
                        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                        [[128, 128, 0], [0, 128, 128], [128, 0, 128]],
                        [[64, 64, 64], [192, 192, 192], [255, 255, 255]],
                    ],
                    dtype=uint8,
                ),
                "xdim": 2 * micro,
                "ydim": 2 * micro,
                "type": "surface",
            },
            ImageData(
                type=ImageType.SURFACE,
                depth_data=array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
                texture_data=array(
                    [
                        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                        [[128, 128, 0], [0, 128, 128], [128, 0, 128]],
                        [[64, 64, 64], [192, 192, 192], [255, 255, 255]],
                    ],
                    dtype=uint8,
                ),
                xdim=2 * micro,
                ydim=2 * micro,
            ),
            id="texture",
        ),
    ),
)
def test_load_mat_file(
    name: str,
    data: dict[str, Any],
    expected: ImageData,
    create_mat_file: Callable[[str, dict[str, Any]], Path],
) -> None:
    path = create_mat_file(name, data)
    actual = _load_mat_file(path)
    assert compare_image_data(actual, expected), (
        f"ImageData mismatch:\nActual: {actual}\nExpected: {expected}"
    )
