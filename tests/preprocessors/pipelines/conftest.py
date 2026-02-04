from pathlib import Path

import pytest
from container_models.image import ImageContainer

from preprocessors.pipelines import parse_scan_pipeline


@pytest.fixture(scope="session")
def _parsed_al3d_file(scan_directory: Path) -> ImageContainer:
    """Parse the circle.al3d test file."""
    return parse_scan_pipeline(scan_directory / "circle.al3d", 1, 1)


@pytest.fixture
def parsed_al3d_file(_parsed_al3d_file: ImageContainer) -> ImageContainer:
    return _parsed_al3d_file.model_copy(deep=True)
