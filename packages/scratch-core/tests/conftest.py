from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def resources_dir() -> Path:
    """Path to resources directory."""
    return TEST_ROOT / "resources"


@pytest.fixture(scope="session")
def scans_dir(resources_dir: Path) -> Path:
    """Path to resources scan sub directory."""
    return resources_dir / "scans"


@pytest.fixture
def image_data(scans_dir: Path) -> np.ndarray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(scans_dir / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data
