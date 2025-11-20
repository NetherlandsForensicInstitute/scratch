from pathlib import Path

from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK

from constants import PROJECT_ROOT


def test_add_scan_placeholder(client: TestClient, tmp_path: Path) -> None:
    # Arrange, We might want to pull resources out to a more global space
    scan_file = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans/circle.al3d"
    output_dir = tmp_path / "output"
    output_dir.mkdir()  # Create output directory

    # Act
    response = client.post(
        "/pre-processor/ingest/",
        json={
            "scan_file": str(scan_file),
            "output_dir": str(output_dir),
        },
    )

    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from add-scan"}, "A placeholder response should be returned"


def test_pre_processors_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/pre-processor")

    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the pre-processors"}, "A placeholder response should be returned"
