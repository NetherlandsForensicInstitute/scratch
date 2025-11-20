from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK


def test_add_scan_placeholder(client: TestClient) -> None:
    # Act
    response = client.post("/add-scan/")
    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from add-scan"}, "A placeholder response should be returned"
