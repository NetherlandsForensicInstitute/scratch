from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
from PIL import Image

from parsers.data_types import ScanImage

TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def scans_dir() -> Path:
    """Path to resources scan directory."""
    return TEST_ROOT / "resources" / "scans"


@pytest.fixture
def image_data(scans_dir: Path) -> NDArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(scans_dir / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture
def scan_image(image_data: NDArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=image_data, path_to_original_image=Path("some/path/file.x3p"))
