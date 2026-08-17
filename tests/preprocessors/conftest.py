from collections.abc import Callable
from pathlib import Path

import pytest

from preprocessors.schemas import EditImage, UploadScan
from tests.constants import CUTOFF_LENGTH


@pytest.fixture(scope="module")
def edit_image_parameter(scan_directory: Path) -> Callable[..., EditImage]:
    def wrapper(**kwargs) -> EditImage:
        return EditImage.model_validate(
            {"scan_file": scan_directory / "circle.x3p", "cutoff_length": CUTOFF_LENGTH, "surface_terms": "none"}
            | kwargs
        )

    return wrapper


@pytest.fixture(scope="module")
def edit_image(scan_directory: Path) -> EditImage:
    return EditImage(scan_file=scan_directory / "circle.x3p", cutoff_length=CUTOFF_LENGTH)  # type: ignore


@pytest.fixture(scope="module")
def upload_scan_parameter(scan_directory: Path) -> Callable[..., UploadScan]:
    def wrapper(**kwargs) -> UploadScan:
        return UploadScan.model_validate({"scan_file": scan_directory / "circle.x3p"} | kwargs)

    return wrapper


@pytest.fixture(scope="module")
def upload_scan(scan_directory: Path) -> UploadScan:
    return UploadScan(scan_file=scan_directory / "circle.x3p")  # type: ignore
