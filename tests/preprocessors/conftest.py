from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

from preprocessors.schemas import EditImage, Mask, UploadScan

MASK: Final[Mask] = ((1, 0, 1), (0, 1, 0))  # type: ignore
CUTOFF_LENGTH: Final[float] = 250


@pytest.fixture(scope="module")
def edit_image_parameter(scan_directory: Path) -> Callable[..., EditImage]:
    def wrapper(**kwargs) -> EditImage:
        return EditImage.model_validate(
            {"scan_file": scan_directory / "circle.x3p", "mask": MASK, "cutoff_length": CUTOFF_LENGTH} | kwargs
        )

    return wrapper


@pytest.fixture(scope="module")
def edit_image(scan_directory: Path) -> EditImage:
    return EditImage(
        scan_file=scan_directory / "circle.x3p",
        mask=MASK,
        cutoff_length=CUTOFF_LENGTH,
    )  # type: ignore


@pytest.fixture(scope="module")
def upload_scan_parameter(scan_directory: Path) -> Callable[..., UploadScan]:
    def wrapper(**kwargs) -> UploadScan:
        return UploadScan.model_validate({"scan_file": scan_directory / "circle.x3p"} | kwargs)

    return wrapper


@pytest.fixture(scope="module")
def upload_scan(scan_directory: Path) -> UploadScan:
    return UploadScan(scan_file=scan_directory / "circle.x3p")  # type: ignore
