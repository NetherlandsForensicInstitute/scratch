from fastapi.testclient import TestClient

from main import app
from settings import get_settings


def test_application_files_are_deleted_after_restart() -> None:
    """Test that temporary files are automatically cleaned up when the application shuts down."""
    # Act
    with TestClient(app) as _:
        # Assert - storage directory should exist during app lifetime
        storage_dir = get_settings().storage
        assert storage_dir.exists()

    # Assert - storage directory should be cleaned up after shutdown (default non-persistent)
    assert not storage_dir.exists()
