from pathlib import Path
import logging

import numpy as np
import pytest
from PIL import Image
from loguru import logger

from container_models.scan_image import ScanImage
from parsers.loaders import load_scan_image
from .helper_function import unwrap_result
from container_models.base import ScanMap2DArray


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
def scan_image_array(baseline_images_dir: Path) -> ScanMap2DArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(baseline_images_dir / "circle.png").convert("L")
    return np.asarray(gray, dtype=np.float64)


@pytest.fixture(scope="session")
def scan_image(scan_image_array: ScanMap2DArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=scan_image_array, scale_x=1, scale_y=1)


@pytest.fixture(scope="session")
def scan_image_replica(scans_dir: Path) -> ScanImage:
    """Build a `ScanImage` object`."""
    return unwrap_result(
        load_scan_image(
            scans_dir / "Klein_non_replica_mode.al3d",
            step_size_x=1,
            step_size_y=1,
        )
    )


@pytest.fixture(scope="session")
def scan_image_with_nans(scan_image_replica: ScanImage) -> ScanImage:
    # add random NaN values
    rng = np.random.default_rng(42)
    scan_image = scan_image_replica.model_copy(deep=True)
    scan_image.data[rng.random(size=scan_image.data.shape) < 0.1] = np.nan
    return scan_image
