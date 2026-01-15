from http import HTTPStatus

from fastapi.testclient import TestClient


def test_pre_processors_placeholder(client: TestClient) -> None:
    """Test that the preprocessor root endpoint redirects to documentation."""
    # Act
    response = client.get("/preprocessor", follow_redirects=False)

    # Assert
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, "endpoint should redirect"
    assert response.headers["location"] == "/docs#operations-tag-preprocessor", "should redirect to preprocessor docs"
