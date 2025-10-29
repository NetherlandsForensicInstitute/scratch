from fastapi.testclient import TestClient


def test_processors_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/processor")
    # Assert
    assert response.status_code == 200, "endpoint is alive"
    assert response.json() == {"message": "Hello from the processors"}, "A placeholder response should be returned"
