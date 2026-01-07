from http import HTTPStatus
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from constants import EXTRACTOR_ROUTE
from models import DirectoryAccess


@pytest.fixture
def written_directory_access(directory_access: DirectoryAccess) -> DirectoryAccess:
    directory_access.resource_path.mkdir(exist_ok=True)
    return directory_access


def test_get_file_returns_file_response(client: TestClient, written_directory_access: DirectoryAccess) -> None:
    """Test that X3P files are served with correct content-type (application/octet-stream)."""
    # Arrange
    filepath = written_directory_access.resource_path / "test.x3p"
    filepath.write_bytes(b"fakeimagecontent")
    # Act
    response = client.get(f"{EXTRACTOR_ROUTE}/files/{written_directory_access.token}/test.x3p")
    # Assert
    assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
    assert response.content == b"fakeimagecontent"
    assert response.headers["content-type"] == "application/octet-stream"


def test_get_file_returns_image_response(client: TestClient, written_directory_access: DirectoryAccess) -> None:
    """Test that PNG files are served with correct content-type (image/png)."""
    # Arrange
    filepath = written_directory_access.resource_path / "test.png"
    filepath.write_bytes(b"fakeimagecontent")
    # Act
    response = client.get(f"{EXTRACTOR_ROUTE}/files/{written_directory_access.token}/test.png")
    # Assert
    assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
    assert response.content == b"fakeimagecontent"
    assert response.headers["content-type"] == "image/png"


def test_get_file_returns_404_for_missing_file(client: TestClient, written_directory_access: DirectoryAccess) -> None:
    """Test that requesting a nonexistent file returns 404 with appropriate error message."""
    # Arrange
    wrong_filename = "nofile.png"
    # Act
    response = client.get(f"{EXTRACTOR_ROUTE}/files/{written_directory_access.token}/{wrong_filename}")
    # Assert
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["detail"] == f"File {wrong_filename} not found."


def test_get_file_rejects_invalid_extension(client: TestClient, directory_access: DirectoryAccess) -> None:
    """Test that requesting a file with an unsupported extension returns 422 validation error."""
    # Act
    response = client.get(f"{EXTRACTOR_ROUTE}/files/{directory_access.token}/test.txt")

    # Assert
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["detail"][0]["msg"] == "Value error, unsupported file type: test.txt, try: x3p, png"


def test_get_file_returns_422_for_invalid_token(client: TestClient) -> None:
    """Test that requesting a file with an invalid token or tag returns 422 validation error."""
    # Arrange
    fake_token = uuid4()

    # Act
    response = client.get(f"{EXTRACTOR_ROUTE}/files/{fake_token}/test.png")

    # Assert
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert f"Unable to fetch resources of token '{fake_token}'" == response.json()["detail"]
