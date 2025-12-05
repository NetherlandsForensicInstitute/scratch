import numpy as np
import pytest
from PIL import Image

from parsers.data_types import ScanImage
from utils.array_definitions import ScanMap2DArray, MaskArray

from .constants import SCANS_DIR


@pytest.fixture
def image_data() -> ScanMap2DArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(SCANS_DIR / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture
def scan_image(image_data: ScanMap2DArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=image_data)


@pytest.fixture
def scan_image_replica() -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage.from_file(SCANS_DIR / "Klein_non_replica_mode.al3d")


@pytest.fixture(scope="module")
def scan_image_with_nans() -> ScanImage:
    """Build a `ScanImage` object`."""
    scan_image = ScanImage.from_file(SCANS_DIR / "Klein_non_replica_mode.al3d")
    # add random NaN values
    rng = np.random.default_rng(42)
    scan_image.data[rng.random(size=scan_image.data.shape) < 0.1] = np.nan
    return scan_image


@pytest.fixture(scope="module")
def mask_array() -> MaskArray:
    """Build a `MaskArray` object`."""
    scan_image = ScanImage.from_file(SCANS_DIR / "Klein_non_replica_mode.al3d")
    data = np.ones_like(scan_image.data).astype(bool)
    # Set the borders (edges) to 0
    data[0, :] = 0  # First row
    data[-1, :] = 0  # Last row
    data[:, 0] = 0  # First column
    data[:, -1] = 0  # Last column
    return data
