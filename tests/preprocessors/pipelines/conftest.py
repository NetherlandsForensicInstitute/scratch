from pathlib import Path

import pytest
from container_models.scan_image import ScanImage

from preprocessors.pipelines import parse_scan_pipeline
from preprocessors.schemas import UploadScanParameters


@pytest.fixture(scope="session")
def default_parameters() -> UploadScanParameters:
    return UploadScanParameters.model_construct()


@pytest.fixture(scope="session")
def parsed_al3d_file(scan_directory: Path, default_parameters: UploadScanParameters) -> ScanImage:
    """Parse the circle.al3d test file."""
    return parse_scan_pipeline(scan_directory / "circle.al3d", default_parameters)
