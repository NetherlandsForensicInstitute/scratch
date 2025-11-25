from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import pytest
from PIL import Image

from parsers.data_types import ScanImage

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
def scan_image(image_data: NDArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=image_data)


@pytest.fixture
def scan_image_replica(scans_dir: Path) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage.from_file(scans_dir / "Klein_non_replica_mode.al3d")


@pytest.fixture
def scan_image_with_nans(scans_dir: Path) -> ScanImage:
    """Build a `ScanImage` object`."""
    scan_image = ScanImage.from_file(scans_dir / "Klein_non_replica_mode.al3d")
    # add random NaN values
    rng = np.random.default_rng(42)
    scan_image.data[rng.random(size=scan_image.data.shape) < 0.1] = np.nan
    return scan_image
