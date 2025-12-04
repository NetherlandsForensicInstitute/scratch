from math import ceil

import numpy as np
import pytest

from conversion import subsample_data
from image_generation.data_formats import ScanImage
from image_generation.translations import ScanMap2DArray
from returns.pipeline import is_successful

from ..constants import BASELINE_IMAGES_DIR, PRECISION


@pytest.mark.parametrize("step_x, step_y", [(1, 1), (10, 10), (25, 25), (25, 50)])
def test_subsample_matches_size(scan_image: ScanMap2DArray, step_x: int, step_y: int):
    # Arrange
    expected_height = ceil(scan_image.shape[0] / step_y)  # rows
    expected_width = ceil(scan_image.shape[1] / step_x)  # columns

    # Act
    result = subsample_data(scan_image, step_x, step_y)

    #  Assert
    assert result.unwrap().shape == (expected_height, expected_width)


def test_scan_map_updates_scales(scan_image: ScanMap2DArray):
    # Arrange
    scale_x = 3
    scale_y = 3
    step_x = 10
    step_y = 10
    input_data = ScanImage(data=scan_image, scale_x=scale_x, scale_y=scale_y)

    # Act
    subsampled = input_data.subsample_data(step_x=step_x, step_y=step_y).unwrap()

    # Assert
    assert subsampled.scale_x == scale_x * step_x
    assert subsampled.scale_y == scale_y * step_y


@pytest.mark.parametrize(
    "step_x, step_y", [(-2, 2), (0, 0), (0, 3), (2, -1), (-1, -1), (1e3, 1e4)]
)
def test_subsample_rejects_incorrect_sizes(
    scan_image: ScanMap2DArray, step_x: int, step_y: int
):
    assert not is_successful(subsample_data(scan_image, step_x, step_y))


def test_subsample_matches_baseline_output(scan_image_replica: ScanImage):
    verified = np.load(BASELINE_IMAGES_DIR / "replica_subsampled.npy")

    result = subsample_data(
        scan_image=scan_image_replica.data, step_size_x=10, step_size_y=15
    )
    assert np.allclose(result.unwrap(), verified, equal_nan=True, atol=PRECISION)


def test_subsample_creates_new_object(scan_image_replica: ScanImage):
    subsampled = subsample_data(
        scan_image=scan_image_replica.data, step_size_x=5, step_size_y=5
    ).unwrap()
    assert id(subsampled) != id(scan_image_replica)
    assert id(subsampled) != id(scan_image_replica.data)
    assert scan_image_replica.data.ctypes.data != subsampled.ctypes.data
