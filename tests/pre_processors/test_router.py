from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK


def test_add_scan_placeholder(client: TestClient) -> None:
    # Act
    response = client.post("/pre-processor/add-scan/")

    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from add-scan"}, "A placeholder response should be returned"


def test_pre_processors_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/pre-processor")

    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the pre-processors"}, "A placeholder response should be returned"
