from http import HTTPStatus

from fastapi.testclient import TestClient


def test_comparison_placeholder(client: TestClient) -> None:
    """Test that the comparator root endpoint redirects to documentation."""
    # Act
    response = client.get("/comparator", follow_redirects=False)
    # Assert
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, "endpoint should redirect"
    assert response.headers["location"] == "/docs#operations-tag-comparator", "should redirect to comparator docs"
