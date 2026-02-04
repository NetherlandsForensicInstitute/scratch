import numpy as np
import pytest

from container_models.base import UnitVector, VectorField
from container_models.image import ImageContainer, MetaData


from ..helper_function import spherical_to_unit_vector

TEST_IMAGE_SIZE = 10
TEST_IMAGE_CENTER = TEST_IMAGE_SIZE // 2


@pytest.fixture(scope="module")
def light_source() -> UnitVector:
    """Single light from 45 degrees azimuth and elevation as UnitVector."""
    return spherical_to_unit_vector(azimuth=45, elevation=45)


@pytest.fixture(scope="module")
def observer() -> UnitVector:
    """Observer looking straight down from +Z direction as UnitVector."""
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


@pytest.fixture(scope="module")
def varied_normals() -> VectorField:
    """Vector field of 3D surface normals (varied orientation)."""
    return np.stack(
        [
            np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.7),  # nx
            np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.6),  # ny
            np.full((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), 0.2),  # nz
        ],
        axis=-1,
    )


@pytest.fixture(scope="module")
def flat_normals() -> VectorField:
    """Vector field with 3D surface normals (all pointing up +Z)."""
    return np.stack(
        [
            np.zeros((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),  # nx
            np.zeros((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),  # ny
            np.ones((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),  # nz
        ],
        axis=-1,
    )


@pytest.fixture
def flat_image(flat_scale: MetaData) -> ImageContainer:
    """ImageContainer with flat (constant) depth data - produces normals pointing +Z."""
    return ImageContainer(
        data=np.ones((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), dtype=np.float64),
        metadata=flat_scale,
    )


@pytest.fixture
def bumpy_image(flat_scale: MetaData) -> ImageContainer:
    """ImageContainer with a bump in the center - produces varied normals."""
    # Create a bump in the center using a gaussian-like shape
    y, x = np.ogrid[:TEST_IMAGE_SIZE, :TEST_IMAGE_SIZE]
    center = TEST_IMAGE_SIZE // 2
    return ImageContainer(
        data=np.exp(-((x - center) ** 2 + (y - center) ** 2) / 8.0), metadata=flat_scale
    )
