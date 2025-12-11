import numpy as np
import pytest

from container_models.light_source import LightSource
from container_models.surface_normals import SurfaceNormals


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
def varied_surface_normals() -> SurfaceNormals:
    """Surface normals with varied orientation."""
    return SurfaceNormals(
        x_normal_vector=np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.7),
        y_normal_vector=np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.6),
        z_normal_vector=np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.2),
    )


@pytest.fixture(scope="module")
def flat_surface_normals() -> SurfaceNormals:
    """Surface normals all pointing straight up (+Z)."""
    return SurfaceNormals(
        x_normal_vector=np.zeros((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),
        y_normal_vector=np.zeros((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),
        z_normal_vector=np.ones((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),
    )
