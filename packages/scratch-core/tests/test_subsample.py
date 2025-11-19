from parsers.data_types import ScanImage
from conversion import subsample_data
import pytest
from math import ceil


@pytest.mark.parametrize("step_size", [1, 10, [25, 25], [25, 50]])
def test_subsample_by_stepsize(scan_image: ScanImage, step_size: int | tuple[int, int]):
    subsampled = subsample_data(scan_image=scan_image, step_size=step_size)
    if isinstance(step_size, int):
        step_size = (step_size, step_size)
    assert subsampled.is_subsampled is True
    assert subsampled.path_to_original_image == scan_image.path_to_original_image
    assert subsampled.width == ceil(scan_image.width / step_size[0])
    assert subsampled.height == ceil(scan_image.height / step_size[1])
    assert subsampled.scale_x == pytest.approx(scan_image.scale_x * step_size[0])
    assert subsampled.scale_y == pytest.approx(scan_image.scale_y * step_size[1])
