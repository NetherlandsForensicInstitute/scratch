import numpy as np
import pytest
from PIL import Image

from image_generation.data_formats import ScanImage
from conversion.data_formats import MarkType, CropType, MarkImage
from parsers.data_types import load_scan_image
from utils.array_definitions import ScanMap2DArray, MaskArray

from .constants import SCANS_DIR


@pytest.fixture(scope="session")
def scan_image_array() -> ScanMap2DArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(SCANS_DIR / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture(scope="session")
def scan_image(scan_image_array: ScanMap2DArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=scan_image_array, scale_x=4e-6, scale_y=4e-6)


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


@pytest.fixture(scope="session")
def mask_array(scan_image) -> MaskArray:
    """Build a `MaskArray` object`."""
    data = np.ones_like(scan_image.data).astype(bool)
    # Set the borders (edges) to 0
    data[0, :] = 0  # First row
    data[-1, :] = 0  # Last row
    data[:, 0] = 0  # First column
    data[:, -1] = 0  # Last column
    return data


@pytest.fixture(scope="session")
def mark_image() -> MarkImage:
    return MarkImage(
        data=np.zeros((100, 100)),
        scale_x=1e-6,
        scale_y=1e-6,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_type=CropType.RECTANGLE,
    )
