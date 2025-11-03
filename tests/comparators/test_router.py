from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK


def test_comparison_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/comparator")
    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the comparator"}, "A placeholder response should be returned"
