"""Tests for preprocessor helper functions."""

from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import HttpUrl

from constants import EXTRACTOR_ROUTE
from file_services import create_vault, fetch_resource_path, get_files, get_urls
from models import DirectoryAccess
from settings import get_settings


@pytest.mark.usefixtures("tmp_dir_api")
class TestFileServices:
    """Tests for preprocessor helper functions."""

    def test_create_vault(self) -> None:
        """Test that create_vault creates a DirectoryAccess with proper directory."""
        # Act
        vault = create_vault("test-tag")

        # Assert
        assert isinstance(vault, DirectoryAccess)
        assert vault.tag == "test-tag"
        assert vault.resource_path.exists()
        assert vault.resource_path.is_dir()

    def test_get_output_files(self, directory_access: DirectoryAccess) -> None:
        """Test that get_output_files returns correct file paths."""
        # Arrange
        directory = get_settings().storage / f"{directory_access.tag}-{directory_access.token.hex}"
        input_files = {"scan": "scan.x3p", "surface_map": "surface_map.png", "preview": "preview.png"}
        expected_paths = {
            "scan": (directory / "scan.x3p"),
            "preview": (directory / "preview.png"),
            "surface_map": (directory / "surface_map.png"),
        }
        # Act
        files = get_files(directory_access.resource_path, **input_files)

        # Assert
        assert files == expected_paths

    def test_get_output_urls(self, directory_access: DirectoryAccess) -> None:
        """Test that get_output_urls returns correct URL mapping."""
        # Arrange
        base_url = f"{get_settings().base_url}{EXTRACTOR_ROUTE}/files/{directory_access.token}"
        expected_urls = dict(
            scan=HttpUrl(f"{base_url}/scan.x3p"),
            preview=HttpUrl(f"{base_url}/preview.png"),
            surface_map=HttpUrl(f"{base_url}/surface_map.png"),
        )
        files = {"scan": "scan.x3p", "surface_map": "surface_map.png", "preview": "preview.png"}
        # Act
        urls = get_urls(directory_access.access_url, **files)

        # Assert
        assert urls == expected_urls

    def test_io_fetch_success(self) -> None:
        """Test that io_fetch successfully reconstructs DirectoryAccess from token."""
        # Arrange
        original = DirectoryAccess(tag="my-project")
        original.resource_path.mkdir(parents=True)

        # Act
        fetched = fetch_resource_path(original.token)

        # Assert
        assert fetched == original.resource_path

    def test_io_fetch_no_directory_found(self) -> None:
        """Test that io_fetch raises error when no directory exists for token."""
        # Act & Assert
        with pytest.raises(HTTPException, match="Unable to fetch resources of token"):
            fetch_resource_path(uuid4())
