from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def scans_dir() -> Path:
    """Path to resources scan directory."""
    return TEST_ROOT / "resources" / "scans"


@pytest.fixture(scope="session")
def png_file(scans_dir: Path) -> Path:
    """Path to a single .png image file."""
    return scans_dir / "circle.png"


@pytest.fixture(scope="session")
def x3p_file(scans_dir: Path) -> Path:
    """Path to a single .x3p scan file."""
    return scans_dir / "circle.x3p"


@pytest.fixture(scope="session")
def al3d_file(scans_dir: Path) -> Path:
    """Path to a single .al3d scan file."""
    return scans_dir / "circle.al3d"


@pytest.fixture
def image_data(scans_dir: Path) -> np.ndarray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(scans_dir / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data
