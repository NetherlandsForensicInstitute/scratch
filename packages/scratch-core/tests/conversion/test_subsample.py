from parsers.data_types import ScanImage, ScanDataKind
from conversion import subsample_data
import pytest
from math import ceil
from pathlib import Path
import numpy as np


@pytest.mark.parametrize("step_size", [1, 10, (25, 25), (25, 50)])
def test_subsample_by_size(
    scan_image: ScanImage, step_size: int | tuple[int, int], atol: float
):
    subsampled = subsample_data(scan_image=scan_image, step_size=step_size)
    if isinstance(step_size, int):
        step_size = (step_size, step_size)
    assert scan_image.data_kind == ScanDataKind.ORIGINAL
    assert subsampled.data_kind == ScanDataKind.SUBSAMPLED
    assert subsampled.path_to_original_image == scan_image.path_to_original_image
    assert subsampled.width == ceil(scan_image.width / step_size[0])
    assert subsampled.height == ceil(scan_image.height / step_size[1])
    assert np.isclose(subsampled.scale_x, scan_image.scale_x * step_size[0], atol=atol)
    assert np.isclose(subsampled.scale_y, scan_image.scale_y * step_size[1], atol=atol)


def test_subsample_by_baseline(
    scan_image_replica: ScanImage, baseline_images_dir: Path, atol: float
):
    subsampled = subsample_data(scan_image=scan_image_replica, step_size=(10, 15))
    verified = ScanImage.from_file(baseline_images_dir / "replica_subsampled.x3p")
    assert np.allclose(
        subsampled.data,
        verified.data,
        equal_nan=True,
        atol=atol,
    )
