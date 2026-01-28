import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from loguru import logger

from container_models.base import ScanMap2DArray, MaskArray
from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType, CropType, Mark
from parsers import load_scan_image
from .helper_function import unwrap_result

TEST_ROOT = Path(__file__).parent


@pytest.fixture
def case_dir() -> Path:
    return TEST_ROOT / "resources" / "preprocess_striation"


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
    return ScanImage(data=scan_image_array, scale_x=4e-6, scale_y=4e-6)


@pytest.fixture(scope="session")
def scan_image_replica(scans_dir: Path) -> ScanImage:
    """Build a `ScanImage` object`."""
    return unwrap_result(
        load_scan_image(
            scans_dir / "Klein_non_replica_mode.al3d",
        )
    )


@pytest.fixture(scope="session")
def scan_image_with_nans(scan_image_replica: ScanImage) -> ScanImage:
    # add random NaN values
    rng = np.random.default_rng(42)
    scan_image = scan_image_replica.model_copy(deep=True)
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
def mask_array(scan_image_replica: ScanImage) -> MaskArray:
    """Build a `MaskArray` object`."""
    data = np.ones_like(scan_image_replica.data).astype(bool)
    # Set the borders (edges) to 0
    data[0, :] = 0  # First row
    data[-1, :] = 0  # Last row
    data[:, 0] = 0  # First column
    data[:, -1] = 0  # Last column
    return data


@pytest.fixture(scope="session")
def mark(scan_image: ScanImage) -> Mark:
    return Mark(
        scan_image=scan_image,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_type=CropType.RECTANGLE,
    )
