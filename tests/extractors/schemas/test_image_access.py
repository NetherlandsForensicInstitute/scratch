from pathlib import Path

from extractors.schemas import ImageAccess


def test_image_access_creation() -> None:
    """Test that ImageAccess can be created with proper fields."""
    # Arrange
    resource_path = Path("/tmp/test_dir")  # noqa
    access_url = "http://localhost:8000/extractor/files/token123"

    # Act
    image_access = ImageAccess(resource_path=resource_path, access_url=access_url)

    # Assert
    assert isinstance(image_access, tuple)
    assert image_access.resource_path == resource_path
    assert image_access.access_url == access_url
