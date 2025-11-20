from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from parsers.data_types import Array2D, ScanImage

TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def scans_dir() -> Path:
    """Path to resources scan directory."""
    return TEST_ROOT / "resources" / "scans"


@pytest.fixture(scope="session")
def baseline_images_dir() -> Path:
    """Path to resources baseline images directory."""
    return TEST_ROOT / "resources" / "baseline_images"


@pytest.fixture(scope="session")
def atol() -> float:
    """Return a small value for the absolute tolerance since parsed values are in meters."""
    return 1e-16


@pytest.fixture
def image_data(scans_dir: Path) -> Array2D:
    """Build a fixture with ground truth image data."""
    gray = Image.open(scans_dir / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture
def scan_image(image_data: Array2D) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=image_data)


@pytest.fixture
def scan_image_replica(scans_dir: Path) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage.from_file(scans_dir / "Klein_non_replica_mode.al3d")
