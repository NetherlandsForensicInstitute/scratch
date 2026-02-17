import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from loguru import logger

from container_models.base import DepthData, BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType, Mark
from conversion.profile_correlator import Profile
from parsers.loaders import load_scan_image
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
def scan_image_array(baseline_images_dir: Path) -> DepthData:
    """Build a fixture with ground truth image data."""
    gray = Image.open(baseline_images_dir / "circle.png").convert("L")
    return np.asarray(gray, dtype=np.float64)


@pytest.fixture(scope="session")
def scan_image(scan_image_array: DepthData) -> ScanImage:
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


@pytest.fixture()
def scan_image_with_nans(scan_image_replica: ScanImage) -> ScanImage:
    # add random NaN values
    rng = np.random.default_rng(42)
    scan_image = scan_image_replica.model_copy(deep=True)
    scan_image.data[rng.random(size=scan_image.data.shape) < 0.1] = np.nan
    return scan_image


@pytest.fixture()
def scan_image_rectangular_with_nans(scan_image_with_nans: ScanImage) -> ScanImage:
    """Build a `ScanImage` object` with non-square image data."""
    scan_image = ScanImage(
        data=scan_image_with_nans.data[:, : scan_image_with_nans.data.shape[1] // 2],
        scale_x=scan_image_with_nans.scale_x * 1.5,
        scale_y=scan_image_with_nans.scale_y,
    )
    return scan_image


@pytest.fixture(scope="module")
def mask_array(scan_image_replica: ScanImage) -> BinaryMask:
    """Build a `MaskArray` object`."""
    data = np.ones_like(scan_image_replica.data).astype(bool)
    # Set the borders (edges) to 0
    data[0, :] = 0  # First row
    data[-1, :] = 0  # Last row
    data[:, 0] = 0  # First column
    data[:, -1] = 0  # Last column
    return data


@pytest.fixture(scope="session")
def impression_mark(scan_image: ScanImage) -> Mark:
    return Mark(
        scan_image=scan_image,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
    )


def striation_mark(profile: Profile, n_cols: int = 50) -> Mark:
    """
    Build a 2D striation Mark by tiling a profile across columns.

    :param profile: Source profile whose heights become the row data.
    :param n_cols: Number of columns in the resulting striation mark.
    :returns: Mark with data shape (len(profile.heights), n_cols).
    """
    data = np.tile(profile.heights[:, np.newaxis], (1, n_cols))
    return Mark(
        scan_image=ScanImage(
            data=data,
            scale_x=profile.pixel_size,
            scale_y=profile.pixel_size,
        ),
        mark_type=MarkType.BULLET_GEA_STRIATION,
    )


@pytest.fixture
def profile_with_nans(pixel_size_05um: float) -> Profile:
    """Create a profile with some NaN values for NaN handling tests."""
    np.random.seed(45)
    x = np.linspace(0, 10 * np.pi, 1000)
    data = np.sin(x) * 1e-6
    data += np.random.normal(0, 0.01e-6, len(data))

    # Insert some NaN values
    data[100:110] = np.nan  # Block of NaNs
    data[500] = np.nan  # Single NaN
    data[700:750] = np.nan  # Larger block

    return Profile(heights=data, pixel_size=pixel_size_05um)


@pytest.fixture
def pixel_size_05um() -> float:
    """Standard pixel size of 0.5 micrometers in meters."""
    return 0.5e-6


@pytest.fixture
def pixel_size_1um() -> float:
    """Standard pixel size of 1.0 micrometer in meters."""
    return 1.0e-6
