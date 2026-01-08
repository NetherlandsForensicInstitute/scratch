import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from loguru import logger

from container_models.base import ScanMap2DArray, MaskArray
from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType, CropType, Mark, CropInfo
from parsers.loaders import load_scan_image
from .helper_function import unwrap_result

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
def masks_dir() -> Path:
    """Path to resources baseline images directory."""
    return TEST_ROOT / "resources" / "masks"


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


@pytest.fixture(scope="session")
def scan_image_before_crop_and_rotate(scans_dir: Path) -> ScanImage:
    """Scan image object to test crop and rotate actions."""
    return unwrap_result(load_scan_image(scans_dir / "9mm_knal.x3p"))


@pytest.fixture(scope="session")
def scan_image_after_crop_and_rotate() -> ScanImage:
    """Scan image object of the result of cropping and rotating actions."""
    pass


@pytest.fixture(scope="session")
def single_rectangle_mask(masks_dir: Path) -> MaskArray:
    return np.load(masks_dir / "single_straight_rectangle_crop_mask.npy")


@pytest.fixture(scope="session")
def multi_shape_mask(masks_dir: Path) -> MaskArray:
    return np.load(masks_dir / "multi_crop_type_mask.npy")


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


@pytest.fixture(scope="session")
def crop_info_single_rectangle() -> list[CropInfo]:
    """
    CropInfo object for single, straight positioned rectangle crop,
    matching mask `./resources/masks/single_straight_rectangle_crop.npy`.
    """
    return [
        CropInfo(
            crop_type=CropType.RECTANGLE,
            data={
                "corner": np.array(
                    [
                        [np.int64(1255), np.int64(1262)],
                        [np.int64(3766), np.int64(1262)],
                        [np.int64(1255), np.int64(3773)],
                        [np.int64(3766), np.int64(3773)],
                    ]
                )
            },
            is_foreground=True,
        )
    ]


@pytest.fixture(scope="session")
def crop_info_multiple_shapes_rectangle_first() -> list[CropInfo]:
    """
    CropInfo object for cropping multiple shapes, matching mask `./resources/masks/multi_crop_type_mask.npy`, with
    the first shape being a rectangle.
    """
    return [
        CropInfo(
            crop_type=CropType.RECTANGLE,
            data={
                "corner": np.array(
                    [
                        [np.int64(278), np.int64(1987)],
                        [np.int64(1426), np.int64(207)],
                        [np.int64(1789), np.int64(442)],
                        [np.int64(641), np.int64(2222)],
                    ]
                )
            },
            is_foreground=True,
        ),
        CropInfo(
            crop_type=CropType.CIRCLE,
            data={
                "center": [np.int64(3050), np.int64(1933)],
                "radius": np.float32(1256.0),
            },
            is_foreground=True,
        ),
        CropInfo(
            crop_type=CropType.ELLIPSE,
            data={
                "center": [np.float32(2245.5), np.float32(2468.5)],
                "minoraxis": np.float64(405.55394215813016),
                "majoraxis": np.float64(1344.2724798194747),
                "angle_majoraxis": np.float64(130.8070195194444),
            },
            is_foreground=False,
        ),
        CropInfo(
            crop_type=CropType.POLYGON,
            data={
                "points": np.array(
                    [
                        [np.int64(2438), np.int64(3530)],
                        [np.int64(3389), np.int64(3627)],
                        [np.int64(3767), np.int64(3769)],
                        [np.int64(2383), np.int64(4252)],
                        [np.int64(1255), np.int64(3769)],
                    ]
                )
            },
            is_foreground=True,
        ),
    ]


@pytest.fixture(scope="session")
def crop_info_multiple_shapes_rectangle_not_first(
    crop_info_multiple_shapes_rectangle_first: list[CropInfo],
) -> list[CropInfo]:
    """
    CropInfo object for crop of multiple shapes, matching mask `./resources/masks/multi_crop_type_mask.npy`, with
    the first shape **not** being a rectangle.
    """
    crop_info_copy = crop_info_multiple_shapes_rectangle_first.copy()
    crop_info_copy.reverse()
    return crop_info_copy


@pytest.fixture(scope="session")
def mark(scan_image: ScanImage, crop_info_single_rectangle: list[CropInfo]) -> Mark:
    return Mark(
        scan_image=scan_image,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_info=crop_info_single_rectangle,
    )
