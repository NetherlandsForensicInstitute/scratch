import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from loguru import logger
from scipy.constants import micro

from container_models.base import DepthData, BinaryMask, Pair, UnitVector
from container_models import ImageContainer
from container_models.image import MetaData, ProcessImage
from .helper_function import spherical_to_unit_vector

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
def scan_image_array(baseline_images_dir: Path) -> DepthData:
    """Build a fixture with ground truth image data."""
    gray = Image.open(baseline_images_dir / "circle.png").convert("L")
    return np.asarray(gray, dtype=np.float64)


@pytest.fixture
def process_image(scan_image_array: DepthData) -> ProcessImage:
    """Build a `ImageContainer` object`."""
    return ProcessImage(
        data=scan_image_array,
        metadata=MetaData(scale=Pair(4 * micro, 4 * micro)),
    )


@pytest.fixture(scope="session")
def _image_replica(scans_dir: Path) -> ProcessImage:
    """Build a `ImageContainer` object`."""
    return ProcessImage.from_scan_file(scans_dir / "Klein_non_replica_mode.al3d")


@pytest.fixture
def image_replica(_image_replica: ProcessImage) -> ProcessImage:
    """Build a `ImageContainer` object`."""
    return _image_replica.model_copy(deep=True)


@pytest.fixture
def image_with_nans(image_replica: ProcessImage) -> ProcessImage:
    # add random NaN values
    rng = np.random.default_rng(42)
    image_replica.data[rng.random(size=image_replica.data.shape) < 0.1] = np.nan
    return image_replica


@pytest.fixture
def image_rectangular_with_nans(
    image_with_nans: ProcessImage,
) -> ImageContainer:
    """Build a `ImageContainer` object` with non-square image data."""
    scale = image_with_nans.metadata.scale
    image_with_nans.data = image_with_nans.data[:, : image_with_nans.width // 2]
    image_with_nans.metadata.scale = Pair(scale.x * 1.5, scale.y)
    return image_with_nans


@pytest.fixture
def mask_array(image_replica: ImageContainer) -> BinaryMask:
    """Build a `MaskArray` object`."""
    data = np.ones_like(image_replica.data).astype(bool)
    # Set the borders (edges) to 0
    data[0, :] = 0  # First row
    data[-1, :] = 0  # Last row
    data[:, 0] = 0  # First column
    data[:, -1] = 0  # Last column
    return data


# Lighting fixtures for shading tests
TEST_IMAGE_SIZE = 10


@pytest.fixture(scope="module")
def light_source() -> UnitVector:
    """Single light from 45 degrees azimuth and elevation as UnitVector."""
    return spherical_to_unit_vector(azimuth=45, elevation=45)


@pytest.fixture(scope="module")
def observer() -> UnitVector:
    """Observer looking straight down from +Z direction as UnitVector."""
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


@pytest.fixture
def flat_scale() -> MetaData:
    return MetaData(scale=Pair(x=1.0, y=1.0))


@pytest.fixture
def flat_image(flat_scale: MetaData) -> ImageContainer:
    """ImageContainer with flat (constant) depth data - produces normals pointing +Z."""
    return ImageContainer(
        data=np.ones((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), dtype=np.float64),
        metadata=flat_scale,
    )


@pytest.fixture
def bumpy_image(flat_scale: MetaData) -> ImageContainer:
    """ImageContainer with a bump in the center - produces varied normals."""
    y, x = np.ogrid[:TEST_IMAGE_SIZE, :TEST_IMAGE_SIZE]
    center = TEST_IMAGE_SIZE // 2
    return ImageContainer(
        data=np.exp(-((x - center) ** 2 + (y - center) ** 2) / 8.0),
        metadata=flat_scale,
    )
