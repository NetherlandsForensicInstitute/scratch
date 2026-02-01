from pathlib import Path

import pytest
from image_tasks.types.scan_image import ScanImage

from preprocessors.new_pipelines import parse_scan_pipeline


@pytest.fixture(scope="session")
def parsed_al3d_file(scan_directory: Path) -> ScanImage:
    """Parse the circle.al3d test file."""
    return parse_scan_pipeline(scan_directory / "circle.al3d", 1, 1)
