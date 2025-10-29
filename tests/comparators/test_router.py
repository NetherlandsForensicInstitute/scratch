from fastapi.testclient import TestClient


def test_comparison_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/comparator")
    # Assert
    assert response.status_code == 200, "endpoint is alive"
    assert response.json() == {"message": "Hello from the comparator"}, "A placeholder response should be returned"
