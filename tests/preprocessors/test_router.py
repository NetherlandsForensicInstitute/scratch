import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from image_generation.exceptions import ImageGenerationError
from parsers.exceptions import ExportError
from pydantic import HttpUrl
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR

from constants import PROJECT_ROOT
from dependencies import get_tmp_dir
from main import app
from preprocessors import ProcessedDataLocation, UploadScan


def test_pre_processors_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/preprocessor")

    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the pre-processors"}, "A placeholder response should be returned"


@pytest.fixture
def tmp_dir_api(
    tmp_path: Path,
) -> Generator[Path, Any, Any]:
    """Replace get_temp_dir to return tmp_path."""
    tmp_dir = tmp_path / uuid4().hex
    tmp_dir.mkdir()
    app.dependency_overrides[get_tmp_dir] = lambda: tmp_dir
    yield tmp_dir
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    app.dependency_overrides.clear()


@pytest.mark.integration
def test_proces_scan(client: TestClient, tmp_dir_api: Path, monkeypatch) -> None:
    # Arrange
    scan_file = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans/circle.x3p"
    input_model = UploadScan(
        scan_file=scan_file,
    )
    fixed_token = "test-token-123"  # noqa: S105
    monkeypatch.setattr(
        "preprocessors.router.uuid4",
        lambda: fixed_token,
    )
    base_url = f"http://localhost:8000/preprocessor/image_file/{fixed_token}"
    # Act
    response = client.post("/preprocessor/process-scan", json=input_model.model_dump(mode="json"))

    # Assert
    expected_response = ProcessedDataLocation(
        x3p_image=HttpUrl(f"{base_url}/scan.x3p"),
        preview_image=HttpUrl(f"{base_url}/preview.png"),
        surfacemap_image=HttpUrl(f"{base_url}/surface_map.png"),
    )

    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    response_model = expected_response.model_validate(response.json())
    assert response_model == expected_response
    assert response_model.x3p_image == HttpUrl(f"{base_url}/scan.x3p")
    assert response_model.preview_image == HttpUrl(f"{base_url}/preview.png")
    assert response_model.surfacemap_image == HttpUrl(f"{base_url}/surface_map.png")
    assert (tmp_dir_api / fixed_token / "scan.x3p").exists()
    assert (tmp_dir_api / fixed_token / "preview.png").exists()
    assert (tmp_dir_api / fixed_token / "surface_map.png").exists()


@pytest.mark.integration
def test_proces_scan_overwrites_files(client: TestClient, tmp_dir_api: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    scan_file = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans/circle.x3p"
    input_model = UploadScan(
        scan_file=scan_file,
    )
    fixed_token = "test-token-123"  # noqa: S105
    monkeypatch.setattr(
        "preprocessors.router.uuid4",
        lambda: fixed_token,
    )
    # Act
    _ = client.post("/preprocessor/process-scan", json=input_model.model_dump(mode="json"))
    x3p_mtime_1 = (tmp_dir_api / fixed_token / "scan.x3p").stat().st_mtime
    preview_mtime_1 = (tmp_dir_api / fixed_token / "preview.png").stat().st_mtime
    client.post("/preprocessor/process-scan", json=input_model.model_dump(mode="json"))

    # Assert
    assert (tmp_dir_api / fixed_token / "scan.x3p").stat().st_mtime > x3p_mtime_1
    assert (tmp_dir_api / fixed_token / "preview.png").stat().st_mtime > preview_mtime_1


@pytest.mark.integration
def test_proces_scan_files_are_deleted_after_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    scan_file = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans/circle.x3p"
    input_model = UploadScan(
        scan_file=scan_file,
    )
    fixed_token = "test-token-123"  # noqa: S105
    monkeypatch.setattr(
        "preprocessors.router.uuid4",
        lambda: fixed_token,
    )
    # Act
    with TestClient(app) as client:
        client.post("/preprocessor/process-scan", json=input_model.model_dump(mode="json"))
        # Assert
        temp_dir = Path(str(app.state.temp_dir.name))
        assert (temp_dir / fixed_token / "preview.png").exists()
    assert not (temp_dir / fixed_token / "preview.png").exists(), "Temp dir should be removed after app shutdown"


@pytest.mark.parametrize(
    ("target_path", "error_kind", "expected_status", "expected_detail"),
    [
        pytest.param(
            "preprocessors.router.save_to_x3p",
            ExportError,
            HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to save the scan file",
            id="save_to_x3p failes",
        ),
        pytest.param(
            "preprocessors.router.get_array_for_display",
            ImageGenerationError,
            HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to generate preview",
            id="Failed to generate preview image",
        ),
        pytest.param(
            "preprocessors.router.compute_3d_image",
            ImageGenerationError,
            HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to generate surface_map",
            id="Failed to generate 3d image",
        ),
    ],
)
@pytest.mark.integration
def test_process_scan_failures(  # noqa
    client: TestClient,
    tmp_dir_api: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_path: str,
    error_kind: type[Exception],
    expected_status: int,
    expected_detail: str,
) -> None:
    # Arrange
    def failing_function(*args, **kwargs) -> None:
        raise error_kind("Test error")

    monkeypatch.setattr(target_path, failing_function)

    scan_file = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans/Klein_non_replica_mode.al3d"
    input_model = UploadScan(scan_file=scan_file)

    # Act
    response = client.post(
        "/preprocessor/process-scan",
        json=input_model.model_dump(mode="json"),
    )

    # Assert
    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]


def test_get_image_returns_file_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_dir_api: Path,
) -> None:
    # Arrange
    fixed_token = "testtoken"  # noqa: S105
    temp_dir = tmp_dir_api / fixed_token
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / "test.x3p"
    file_path.write_bytes(b"fakeimagecontent")
    monkeypatch.setattr(
        "preprocessors.router.uuid4",
        lambda: fixed_token,
    )
    # Act
    response = client.get(f"/preprocessor/image_file/{fixed_token}/test.x3p")
    # Assert
    assert response.status_code == HTTP_200_OK, f"endpoint is alive, {response.text}"
    assert response.content == b"fakeimagecontent"
    assert response.headers["content-type"] == "application/octet-stream"


def test_get_image_returns_image_response(
    client: TestClient, tmp_dir_api: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    fixed_token = "testtoken"  # noqa: S105
    temp_dir = tmp_dir_api / fixed_token
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / "test.png"
    file_path.write_bytes(b"fakeimagecontent")
    monkeypatch.setattr(
        "preprocessors.router.uuid4",
        lambda: fixed_token,
    )
    # Act
    response = client.get(f"/preprocessor/image_file/{fixed_token}/test.png")
    # Assert
    assert response.status_code == HTTP_200_OK, f"endpoint is alive, {response.text}"
    assert response.content == b"fakeimagecontent"
    assert response.headers["content-type"] == "image/png"


def test_get_image_returns_404_for_missing_file(
    client: TestClient, tmp_dir_api: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_token = "testtoken"  # noqa: S105
    temp_dir = tmp_dir_api / fixed_token
    temp_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "preprocessors.router.uuid4",
        lambda: fixed_token,
    )
    wrong_file_name = "nofile.png"
    response = client.get(f"/preprocessor/image_file/{fixed_token}/{wrong_file_name}")
    assert response.status_code == HTTP_404_NOT_FOUND
    assert response.json()["detail"] == f"File {wrong_file_name} not found in temp dir."


def test_get_image_returns_404_for_wrong_token(
    client: TestClient, tmp_dir_api: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    fixed_token = "correcttoken"  # noqa: S105
    temp_dir = tmp_dir_api / fixed_token
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / "test.png"
    file_path.write_bytes(b"data")
    monkeypatch.setattr(
        "preprocessors.router.uuid4",
        lambda: fixed_token,
    )
    # Act: use wrong token
    wrong_token = "wrongtoken"  # noqa: S105
    response = client.get(f"/preprocessor/image_file/{wrong_token}/test.png")

    # Assert
    assert response.status_code == HTTP_404_NOT_FOUND
    assert response.json()["detail"] == f"Temp dir {tmp_dir_api / wrong_token} not found."
