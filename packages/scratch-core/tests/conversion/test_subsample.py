from parsers.data_types import ScanImage
from conversion import subsample_data
import pytest
from math import ceil
import numpy as np
from ..constants import PRECISION, BASELINE_IMAGES_DIR  # type: ignore


@pytest.mark.parametrize("step_size", [(1, 1), (10, 10), (25, 25), (25, 50)])
def test_subsample_matches_size(scan_image: ScanImage, step_size: tuple[int, int]):
    subsampled = subsample_data(scan_image=scan_image, step_size=step_size)
    assert subsampled.width == ceil(scan_image.width / step_size[0])
    assert subsampled.height == ceil(scan_image.height / step_size[1])
    assert np.isclose(
        subsampled.scale_x, scan_image.scale_x * step_size[0], atol=PRECISION
    )
    assert np.isclose(
        subsampled.scale_y, scan_image.scale_y * step_size[1], atol=PRECISION
    )


@pytest.mark.parametrize(
    "step_size", [(-2, 2), (0, 0), (0, 3), (2, -1), (-1, -1), (1e3, 1e4)]
)
def test_subsample_rejects_incorrect_sizes(
    scan_image: ScanImage, step_size: tuple[int, int]
):
    with pytest.raises(ValueError):
        _ = subsample_data(scan_image=scan_image, step_size=step_size)


def test_subsample_matches_baseline_output(scan_image_replica: ScanImage):
    verified = np.load(BASELINE_IMAGES_DIR / "replica_subsampled.npy")

    subsampled = subsample_data(scan_image=scan_image_replica, step_size=(10, 15))
    assert np.allclose(
        subsampled.data,
        verified,
        equal_nan=True,
        atol=PRECISION,
    )


def test_subsample_creates_new_object(scan_image_replica: ScanImage):
    subsampled = subsample_data(scan_image=scan_image_replica, step_size=(5, 5))
    assert id(subsampled) != id(scan_image_replica)
    assert id(subsampled.data) != id(scan_image_replica.data)
    assert scan_image_replica.data.ctypes.data != subsampled.data.ctypes.data
