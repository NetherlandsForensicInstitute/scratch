from container_models.image import ImageContainer
from conversion.container_models import ScanImage
import pytest

from conversion.container_models.base import DepthData
from conversion.data_formats import Mark, MarkType


def _image_container_to_scan_image(image: ImageContainer) -> ScanImage:
    scale = image.metadata.scale
    return ScanImage(data=image.data, scale_x=scale.x, scale_y=scale.y)  # type: ignore


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


@pytest.fixture
def scan_image_replica(image_replica: ImageContainer) -> ScanImage:
    return _image_container_to_scan_image(image_replica)


@pytest.fixture
def scan_image_with_nans(image_with_nans: ImageContainer) -> ScanImage:
    return _image_container_to_scan_image(image_with_nans)


@pytest.fixture
def scan_image_rectangular_with_nans(
    image_rectangular_with_nans: ImageContainer,
) -> ScanImage:
    return _image_container_to_scan_image(image_rectangular_with_nans)
