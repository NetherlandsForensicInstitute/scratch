import pytest
from numpy.typing import NDArray
import numpy as np

N_POINTS = 100


@pytest.fixture
def xs() -> NDArray[np.float64]:
    return np.linspace(-100, 100, num=N_POINTS)


@pytest.fixture
def ys() -> NDArray[np.float64]:
    return np.linspace(-50, 50, num=N_POINTS)


@pytest.fixture
def zs() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return rng.random(size=N_POINTS)
