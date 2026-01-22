from collections.abc import Callable
from pathlib import Path

import pytest

from preprocessors.schemas import UploadScan


@pytest.fixture(scope="module")
def upload_scan_parameter(scan_directory: Path) -> Callable[..., UploadScan]:
    def wrapper(**kwargs) -> UploadScan:
        return UploadScan.model_validate({"scan_file": scan_directory / "circle.x3p"} | kwargs)

    return wrapper


@pytest.fixture(scope="module")
def upload_scan(scan_directory: Path) -> UploadScan:
    return UploadScan(scan_file=scan_directory / "circle.x3p")  # type: ignore
