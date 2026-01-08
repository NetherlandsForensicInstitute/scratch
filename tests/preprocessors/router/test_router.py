from http import HTTPStatus

from fastapi.testclient import TestClient


def test_pre_processors_placeholder(client: TestClient) -> None:
    """Test that the preprocessor root endpoint returns a placeholder message."""
    # Act
    response = client.get("/preprocessor")

    # Assert
    assert response.status_code == HTTPStatus.OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the pre-processors"}, "A placeholder response should be returned"
