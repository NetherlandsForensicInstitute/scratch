from pathlib import Path
import logging

import numpy as np
import pytest
from PIL import Image
from loguru import logger

from image_generation.data_formats import ScanImage
from parsers.data_types import load_scan_image
from utils.array_definitions import ScanMap2DArray, MaskArray

from .constants import SCANS_DIR

TEST_ROOT = Path(__file__).parent


class PropagateHandler(logging.Handler):
    """Handler that propagates loguru records to standard logging."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


@pytest.fixture
def caplog(caplog):
    """Fixture to enable caplog to capture loguru logs."""
    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)


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
def scan_image_rectangular_with_nans(scan_image_with_nans: ScanImage) -> ScanImage:
    """Build a `ScanImage` object` with non-square image data."""
    scan_image = ScanImage(
        data=scan_image_with_nans.data[:, : scan_image_with_nans.data.shape[1] // 2],
        scale_x=scan_image_with_nans.scale_x * 1.5,
        scale_y=scan_image_with_nans.scale_y,
    )
    return scan_image


@pytest.fixture(scope="module")
def mask_array(scan_image_replica) -> MaskArray:
    """Build a `MaskArray` object`."""
    data = np.ones_like(scan_image_replica.data).astype(bool)
    # Set the borders (edges) to 0
    data[0, :] = 0  # First row
    data[-1, :] = 0  # Last row
    data[:, 0] = 0  # First column
    data[:, -1] = 0  # Last column
    return data
