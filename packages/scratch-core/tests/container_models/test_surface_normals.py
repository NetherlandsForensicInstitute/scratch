import numpy as np
from pydantic import ValidationError
import pytest

from container_models.surface_normals import SurfaceNormals


@pytest.mark.parametrize(
    "nx, ny, nz",
    [
        pytest.param((100, 100), (80, 100), (100, 100), id="ny shorter width"),
        pytest.param((100, 100), (100, 80), (100, 100), id="ny shorter height"),
        pytest.param((80, 100), (100, 100), (100, 100), id="nx shorter width"),
        pytest.param((100, 80), (100, 100), (100, 100), id="nx shorter height"),
        pytest.param((100, 100), (100, 100), (80, 100), id="nz shorter width"),
        pytest.param((100, 100), (100, 100), (100, 80), id="nz shorter height"),
    ],
)
def test_surface_normals_invalid_shapes(
    nx: tuple[int, int], ny: tuple[int, int], nz: tuple[int, int]
):
    # act and assert
    with pytest.raises(
        ValidationError,
        match=r"All normal vector components must have the same shape",
    ):
        SurfaceNormals(
            x_normal_vector=np.zeros(nx),
            y_normal_vector=np.zeros(ny),
            z_normal_vector=np.zeros(nz),
        )
