import numpy as np
import pytest

from container_models.light_source import LightSource
from container_models.scan_image import ScanImage


TEST_IMAGE_SIZE = 10
TEST_IMAGE_CENTER = TEST_IMAGE_SIZE // 2


@pytest.fixture(scope="module")
def light_source() -> LightSource:
    """Single light from 45 degrees azimuth and elevation."""
    return LightSource(azimuth=45, elevation=45)


@pytest.fixture(scope="module")
def observer() -> LightSource:
    """Observer looking straight down from +Z direction."""
    return LightSource(azimuth=0, elevation=90)


@pytest.fixture(scope="module")
def varied_normals_scan_image() -> ScanImage:
    """ScanImage with 3D surface normals (varied orientation)."""
    return ScanImage(
        data=np.stack(
            [
                np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.7),  # nx
                np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.6),  # ny
                np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.2),  # nz
            ],
            axis=-1,
        ),
        scale_x=1.0,
        scale_y=1.0,
    )


@pytest.fixture(scope="module")
def flat_normals_scan_image() -> ScanImage:
    """ScanImage with 3D surface normals (all pointing up +Z)."""
    return ScanImage(
        data=np.stack(
            [
                np.zeros((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),  # nx
                np.zeros((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),  # ny
                np.ones((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),  # nz
            ],
            axis=-1,
        ),
        scale_x=1.0,
        scale_y=1.0,
    )
