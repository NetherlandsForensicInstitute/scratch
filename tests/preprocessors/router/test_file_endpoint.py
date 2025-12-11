from http import HTTPStatus
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from preprocessors.router import ROUTE


def test_get_file_returns_file_response(client: TestClient, token: UUID, tmp_dir_api: Path) -> None:
    """Test that X3P files are served with correct content-type (application/octet-stream)."""
    # Arrange
    temp_dir = tmp_dir_api / str(token)
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / "test.x3p"
    file_path.write_bytes(b"fakeimagecontent")
    # Act
    response = client.get(f"{ROUTE}/file/{token}/test.x3p")
    # Assert
    assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
    assert response.content == b"fakeimagecontent"
    assert response.headers["content-type"] == "application/octet-stream"


def test_get_file_returns_image_response(client: TestClient, tmp_dir_api: Path, token: UUID) -> None:
    """Test that PNG files are served with correct content-type (image/png)."""
    # Arrange
    temp_dir = tmp_dir_api / str(token)
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / "test.png"
    file_path.write_bytes(b"fakeimagecontent")
    # Act
    response = client.get(f"{ROUTE}/file/{token}/test.png")
    # Assert
    assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
    assert response.content == b"fakeimagecontent"
    assert response.headers["content-type"] == "image/png"


def test_get_file_returns_404_for_missing_file(client: TestClient, tmp_dir_api: Path, token: UUID) -> None:
    """Test that requesting a nonexistent file returns 404 with appropriate error message."""
    temp_dir = tmp_dir_api / str(token)
    temp_dir.mkdir(exist_ok=True)
    wrong_file_name = "nofile.png"
    response = client.get(f"{ROUTE}/file/{token}/{wrong_file_name}")
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["detail"] == f"File {wrong_file_name} not found."


def test_get_file_returns_400_for_wrong_token(client: TestClient, tmp_dir_api: Path) -> None:
    """Test that requesting a file with an invalid token returns 404 with appropriate error message."""
    # Arrange
    wrong_token = uuid4()

    # Act: use wrong token
    response = client.get(f"{ROUTE}/file/{wrong_token}/test.png")

    # Assert
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] == f"Expired or invalid token: {wrong_token}"


def test_get_file_rejects_invalid_extension(client: TestClient, token: UUID, tmp_dir_api: Path) -> None:
    """Test that requesting a file with an unsupported extension returns 422 validation error."""
    # Arrange
    temp_dir = tmp_dir_api / str(token)
    temp_dir.mkdir(exist_ok=True)

    # Act
    response = client.get(f"{ROUTE}/file/{token}/test.txt")

    # Assert
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["detail"][0]["msg"] == "Value error, File must have one of these extensions: x3p, png"
