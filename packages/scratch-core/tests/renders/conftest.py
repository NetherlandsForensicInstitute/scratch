import numpy as np
import pytest

from container_models.base import VectorField

TEST_IMAGE_SIZE = 10


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
