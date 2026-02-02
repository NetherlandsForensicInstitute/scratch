from conversion.container_models import ScanImage
import pytest

from conversion.container_models.base import DepthData
from conversion.data_formats import Mark, MarkType


@pytest.fixture(scope="session")
def scan_image(scan_image_array: DepthData) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=scan_image_array, scale_x=4e-6, scale_y=4e-6)


@pytest.fixture(scope="session")
def mark(scan_image: ScanImage) -> Mark:
    return Mark(
        scan_image=scan_image,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
    )
