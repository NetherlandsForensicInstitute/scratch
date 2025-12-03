from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from parsers.exceptions import ExportError
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR

from constants import PROJECT_ROOT
from preprocessors import ProcessedDataLocation, UploadScan


def test_pre_processors_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/preprocessor")

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

    # Act
    response = client.post("/preprocessor/process-scan", json=input_model.model_dump(mode="json"))

    # Assert
    expected_response = ProcessedDataLocation(
        preview_image=input_model.output_dir / "preview.png",
        surfacemap_image=input_model.output_dir / "surface_map.png",
        x3p_image=input_model.output_dir / "scan.x3p",
    )
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    response_model = expected_response.model_validate(response.json())
    assert response_model == expected_response


@pytest.mark.parametrize(
    ("target_path", "error_kind", "expected_status", "expected_detail"),
    [
        pytest.param(
            "preprocessors.router.from_file",
            ExportError,
            HTTP_400_BAD_REQUEST,
            "Failed to parse the given scan file, err:Test error",
            id="from_file failes",
        ),
        pytest.param(
            "preprocessors.router.save_to_x3p",
            ExportError,
            HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to save the scan file",
            id="save_to_x3p failes",
        ),
        pytest.param(
            "image_generation.image_generation.get_array_for_display",
            ValueError,
            HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to generate surface_map.png",
            id="generate_3d_image failes",
        ),
    ],
)
@pytest.mark.integration
def test_process_scan_failures(  # noqa
    client: TestClient,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    target_path: str,
    error_kind: type[Exception],
    expected_status: int,
    expected_detail: str,
) -> None:
    # Arrange
    def failing_function(*args, **kwargs) -> None:
        raise error_kind("Test error")

    monkeypatch.setattr(target_path, failing_function)

    scan_file = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans/circle.x3p"
    input_model = UploadScan(scan_file=scan_file, output_dir=tmp_path)

    # Act
    response = client.post(
        "/preprocessor/process-scan",
        json=input_model.model_dump(mode="json"),
    )

    # Assert
    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]
