from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK

from constants import PROJECT_ROOT
from pre_processors.schemas import ProcessScan, UploadScan


def test_pre_processors_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/pre-processor")

    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the pre-processors"}, "A placeholder response should be returned"


@pytest.mark.integration
def test_proces_scan(client: TestClient, tmp_path: Path) -> None:
    # Arrange
    scan_file = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans/circle.x3p"
    input_model = UploadScan(
        scan_file=scan_file,
        output_dir=tmp_path,
    )
    # TODO this will make the files till we implemented the endpoint to bypass the validations
    (input_model.output_dir / "circle.x3p").touch()
    (input_model.output_dir / "preview.png").touch()
    (input_model.output_dir / "surfacemap.png").touch()

    expected_response = ProcessScan(
        preview_image=input_model.output_dir / "preview.png",
        surfacemap_image=input_model.output_dir / "surfacemap.png",
        x3p_image=input_model.output_dir / "circle.x3p",
    )

    # Act
    response = client.post("/pre-processor/process-scan", json=input_model.model_dump(mode="json"), timeout=5)

    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    response_model = expected_response.model_validate(response.json())
    assert response_model == expected_response
