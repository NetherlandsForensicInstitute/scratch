import numpy as np
import pytest

from image_generation.data_formats import SurfaceNormals


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
    nx_array = np.zeros(nx)
    ny_array = np.zeros(ny)
    nz_array = np.zeros(nz)

    with pytest.raises(ValueError) as excinfo:
        SurfaceNormals(np.stack([nx_array, ny_array, nz_array], axis=-1))
    assert "all input arrays must have the same shape" == excinfo.value.args[0]
