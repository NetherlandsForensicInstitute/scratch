import numpy as np
import pytest
from PIL import Image

from image_generation.data_formats import ScanImage
from parsers.data_types import load_scan_image
from utils.array_definitions import ScanMap2DArray

from .constants import SCANS_DIR


@pytest.fixture(scope="session")
def scan_image() -> ScanMap2DArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(SCANS_DIR / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture(scope="session")
def scan_map_2d(scan_image: ScanMap2DArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=scan_image, scale_x=1, scale_y=1)


@pytest.fixture(scope="session")
def scan_image_replica() -> ScanImage:
    """Build a `ScanImage` object`."""
    return load_scan_image(scan_file=SCANS_DIR / "Klein_non_replica_mode.al3d")


@pytest.fixture(scope="session")
def scan_image_with_nans(scan_image_replica: ScanImage) -> ScanImage:
    """Build a `ScanImage` object`."""
    scan_image = scan_image_replica.model_copy(deep=True)
    # add random NaN values
    rng = np.random.default_rng(42)
    scan_image.data[rng.random(size=scan_image.data.shape) < 0.1] = np.nan
    return scan_image
