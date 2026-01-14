from http import HTTPStatus

from fastapi.testclient import TestClient


def test_extractor_placeholder(client: TestClient) -> None:
    """Test that the extractor root endpoint redirects to documentation."""
    # Act
    response = client.get("/extractor", follow_redirects=False)

    # Assert
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, "endpoint should redirect"
    assert response.headers["location"] == "/docs#operations-tag-extractor", "should redirect to extractor docs"
