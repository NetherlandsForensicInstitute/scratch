from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from image_generation.data_formats import ScanMap2D
from image_generation.translations import ScanMap2DArray
from parsers.data_types import from_file

TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def scans_dir() -> Path:
    """Path to resources scan directory."""
    return TEST_ROOT / "resources" / "scans"


@pytest.fixture(scope="session")
def baseline_images_dir() -> Path:
    """Path to resources baseline images directory."""
    return TEST_ROOT / "resources" / "baseline_images"


@pytest.fixture
def image_data(scans_dir: Path) -> NDArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(scans_dir / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture
def scan_image(image_data: NDArray) -> ScanMap2DArray:
    """Build a `ScanImage` object`."""
    return image_data


@pytest.fixture
def scan_map_2d(image_data: NDArray) -> ScanMap2D:
    """Build a `ScanImage` object`."""
    return ScanMap2D(data=image_data)


@pytest.fixture
def scan_image_replica(scans_dir: Path) -> ScanMap2D:
    """Build a `ScanImage` object`."""
    return from_file(scan_file=scans_dir / "Klein_non_replica_mode.al3d")
