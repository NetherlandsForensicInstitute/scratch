from pathlib import Path
import logging

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from loguru import logger
from returns.io import IOSuccess

from image_generation.data_formats import ScanImage
from parsers.data_types import from_file
from utils.array_definitions import ScanMap2DArray
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
def scans_dir() -> Path:
    """Path to resources scan directory."""
    return TEST_ROOT / "resources" / "scans"


@pytest.fixture(scope="session")
def baseline_images_dir() -> Path:
    """Path to resources baseline images directory."""
    return TEST_ROOT / "resources" / "baseline_images"


@pytest.fixture(scope="session")
def image_data() -> ScanMap2DArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(SCANS_DIR / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture(scope="session")
def scan_image(image_data: ScanMap2DArray) -> ScanMap2DArray:
    """Build a `ScanImage` object`."""
    return image_data


@pytest.fixture(scope="session")
def scan_map_2d(image_data: ScanMap2DArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=image_data)


@pytest.fixture(scope="session")
def scan_image_replica() -> ScanImage:
    """Build a `ScanImage` object`."""
    match from_file(scan_file=SCANS_DIR / "Klein_non_replica_mode.al3d"):
        case IOSuccess(image):
            return image.unwrap()
        case _:
            raise Exception


@pytest.fixture(scope="module")
def scan_image_with_nans(scan_image_replica: ScanImage) -> ScanImage:
    # add random NaN values
    rng = np.random.default_rng(42)
    scan_image = scan_image_replica.model_copy()
    scan_image.data[rng.random(size=scan_image.data.shape) < 0.1] = np.nan
    return scan_image
