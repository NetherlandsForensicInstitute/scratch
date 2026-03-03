import numpy as np
import pytest
from container_models.base import FloatArray1D

N_POINTS = 100


@pytest.fixture
def xs() -> FloatArray1D:
    return np.linspace(-100, 100, num=N_POINTS)


@pytest.fixture
def ys() -> FloatArray1D:
    return np.linspace(-50, 50, num=N_POINTS)


@pytest.fixture
def zs(rng: np.random.Generator) -> FloatArray1D:
    return rng.random(size=N_POINTS)
