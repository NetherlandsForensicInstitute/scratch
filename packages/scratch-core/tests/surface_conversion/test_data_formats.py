import numpy as np
import pytest

from surface_conversion.data_formats import SurfaceNormals


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
    nx = np.zeros(nx)
    ny = np.zeros(ny)
    nz = np.zeros(nz)

    with pytest.raises(ValueError) as excinfo:
        SurfaceNormals(nx=nx, ny=ny, nz=nz)

    assert "matching shapes" in str(excinfo.value)
