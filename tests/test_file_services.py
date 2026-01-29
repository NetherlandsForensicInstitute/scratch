"""Tests for preprocessor helper functions."""

from uuid import uuid4

import pytest
from fastapi import HTTPException

from file_services import create_vault, fetch_resource_path
from models import DirectoryAccess


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
