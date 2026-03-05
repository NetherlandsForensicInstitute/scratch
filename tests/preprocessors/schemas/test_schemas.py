import pytest
from pydantic import HttpUrl

from constants import RoutePrefix
from extractors.constants import ProcessFiles
from extractors.schemas import ProcessedDataAccess
from models import DirectoryAccess
from settings import get_settings


@pytest.mark.usefixtures("tmp_dir_api")
def test_get_output_files(directory_access: DirectoryAccess) -> None:
    """Test that get_output_files returns correct file paths."""
    # Arrange
    directory = get_settings().storage / f"{directory_access.tag}-{directory_access.token.hex}"
    expected_paths = (
        directory / "scan.x3p",
        directory / "preview.png",
        directory / "surface_map.png",
    )
    # Act
    files = set(file.get_file_path(directory_access.resource_path) for file in ProcessFiles)

    # Assert
    assert files == set(expected_paths)


@pytest.mark.usefixtures("tmp_dir_api")
def test_get_output_urls(directory_access: DirectoryAccess) -> None:
    """Test that get_output_urls returns correct URL mapping."""
    # Arrange
    base_url = f"{get_settings().base_url}/{RoutePrefix.EXTRACTOR}/files/{directory_access.token}"
    expected_urls = (
        f"{base_url}/scan.x3p",
        f"{base_url}/preview.png",
        f"{base_url}/surface_map.png",
    )
    # Act
    urls = set(file.generate_url(directory_access.access_url) for file in ProcessFiles)

    # Assert
    assert urls == set(expected_urls)
