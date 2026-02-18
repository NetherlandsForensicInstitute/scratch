import pytest
from pydantic import HttpUrl

from constants import RoutePrefix
from extractors.schemas import ProcessedDataAccess
from models import DirectoryAccess
from settings import get_settings


@pytest.mark.usefixtures("tmp_dir_api")
def test_get_output_files(directory_access: DirectoryAccess) -> None:
    """Test that get_output_files returns correct file paths."""
    # Arrange
    directory = get_settings().storage / f"{directory_access.tag}-{directory_access.token.hex}"
    expected_paths = {
        "scan": directory / "scan.x3p",
        "preview": directory / "preview.png",
        "surface_map": directory / "surface_map.png",
    }
    # Act
    files = ProcessedDataAccess.get_files(directory_access.resource_path)

    # Assert
    assert files == expected_paths


@pytest.mark.usefixtures("tmp_dir_api")
def test_get_output_urls(directory_access: DirectoryAccess) -> None:
    """Test that get_output_urls returns correct URL mapping."""
    # Arrange
    base_url = f"{get_settings().base_url}/{RoutePrefix.EXTRACTOR}/files/{directory_access.token}"
    expected_urls = ProcessedDataAccess(
        scan=HttpUrl(f"{base_url}/scan.x3p"),
        preview=HttpUrl(f"{base_url}/preview.png"),
        surface_map=HttpUrl(f"{base_url}/surface_map.png"),
    )
    # Act
    urls = ProcessedDataAccess.generate_urls(directory_access.access_url)

    # Assert
    assert urls == expected_urls
