import numpy as np
import pytest
from PIL import Image

from image_generation.data_formats import ScanImage
from image_generation.translations import ScanMap2DArray
from parsers.data_types import from_file
from utils.array_definitions import ScanMap2DArray

from .constants import SCANS_DIR


@pytest.fixture
def image_data() -> ScanMap2DArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(SCANS_DIR / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture
def scan_image(image_data: ScanMap2DArray) -> ScanMap2DArray:
    """Build a `ScanImage` object`."""
    return image_data


@pytest.fixture
def scan_map_2d(image_data: ScanMap2DArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=image_data)


@pytest.fixture
def scan_image_replica() -> ScanImage:
    """Build a `ScanImage` object`."""
    return from_file(scan_file=SCANS_DIR / "Klein_non_replica_mode.al3d")

@pytest.fixture(scope="module")
def scan_image_with_nans() -> ScanMap2D:
    """Build a `ScanImage` object`."""
    scan_image = from_file(SCANS_DIR / "Klein_non_replica_mode.al3d")
    # add random NaN values
    rng = np.random.default_rng(42)
    scan_image.data[rng.random(size=scan_image.data.shape) < 0.1] = np.nan
    return scan_image
