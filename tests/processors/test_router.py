from fastapi.testclient import TestClient


def test_processors_placeholder(client: TestClient) -> None:
    # Arrange
    response_code_all_good = 200
    # Act
    response = client.get("/processor")
    # Assert
    assert response.status_code == response_code_all_good, "endpoint is alive"
    assert response.json() == {"message": "Hello from the processors"}, "A placeholder response should be returned"
