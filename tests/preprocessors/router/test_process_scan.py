from collections.abc import Callable
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from container_models.light_source import LightSource
from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image
from pydantic import HttpUrl

from constants import PreprocessorEndpoint, RoutePrefix
from extractors.schemas import ProcessedDataAccess
from models import DirectoryAccess
from preprocessors.schemas import UploadScan
from settings import get_settings

PROCESS_SCAN_ROUTE = f"/{RoutePrefix.PREPROCESSOR}/{PreprocessorEndpoint.PROCESS_SCAN}"


@pytest.fixture(scope="module")
def upload_scan(scan_directory: Path) -> UploadScan:
    """Fixture that provides a default UploadScan model using circle.al3d."""
    return UploadScan(scan_file=scan_directory / "Klein_non_replica_mode.al3d")  # type: ignore


@pytest.fixture
def post_process_scan(client: TestClient, upload_scan: UploadScan) -> Callable[[UploadScan | None], Response]:
    """Fixture that provides a function to post to the process-scan endpoint.

    Uses upload_scan (AL3D) by default, but can accept a custom UploadScan model.
    """

    def _post(input_model: UploadScan | None = None) -> Response:
        return client.post(PROCESS_SCAN_ROUTE, json=(input_model or upload_scan).model_dump(mode="json"))

    return _post


@pytest.mark.e2e
@pytest.mark.usefixtures("tmp_dir_api")
class TestProcessScanEndpoint:
    """End-to-end tests for the /process-scan endpoint."""

    def test_process_scan_success_with_al3d_file(self, upload_scan: UploadScan, client: TestClient) -> None:
        """Test successful scan processing with AL3D input file."""
        # Act I
        response = client.post(PROCESS_SCAN_ROUTE, json=upload_scan.model_dump(mode="json"))

        # Assert - verify response
        assert response.status_code == HTTPStatus.OK
        result = ProcessedDataAccess.model_validate(response.json())

        # Act II
        downloads = (client.get(str(url)) for _, url in result)

        # Assert - verify response status codes
        assert all(download.status_code == HTTPStatus.OK for download in downloads)

    def test_process_scan_with_custom_light_sources(
        self, post_process_scan, upload_scan: UploadScan, client: TestClient
    ) -> None:
        """Test that custom light sources produce different surface maps with varying brightness."""
        # Arrange
        single_light = LightSource(azimuth=90, elevation=45)
        observer = LightSource(azimuth=0, elevation=90)

        # Control: one light source
        control_data = UploadScan(  # type:ignore
            scan_file=upload_scan.scan_file,
            light_sources=(single_light,),
            observer=observer,
        )

        # Result: same light source doubled
        request_data = UploadScan(  # type: ignore
            scan_file=upload_scan.scan_file,
            light_sources=(single_light, LightSource(azimuth=180, elevation=45)),
            observer=observer,
        )

        # Act I
        control_response = post_process_scan(control_data)
        response = post_process_scan(request_data)

        # Assert
        assert control_response.status_code == HTTPStatus.OK
        assert response.status_code == HTTPStatus.OK
        result_location = ProcessedDataAccess.model_validate(response.json())
        control_location = ProcessedDataAccess.model_validate(control_response.json())

        # Act II
        control = client.get(str(control_location.surface_map_image))
        result = client.get(str(result_location.surface_map_image))

        # Assert - verify that the two surface_maps are not the same
        assert control.status_code == HTTPStatus.OK
        assert result.status_code == HTTPStatus.OK
        assert control.content != result.content, "Surfacemaps should differ with different light sources"

        # Calculate average brightness for each image
        control_brightness = np.array(Image.open(BytesIO(control.content))).mean()
        result_brightness = np.array(Image.open(BytesIO(result.content))).mean()

        assert result_brightness > control_brightness, (
            f"Result brightness ({result_brightness:.2f}) with two light sources should be greater than "
            f"control brightness ({control_brightness:.2f}) with one light source"
        )


@pytest.mark.usefixtures("tmp_dir_api")
@pytest.mark.integration
class TestProcessScan:
    """Integration tests for the /process-scan endpoint with file system operations."""

    def test_process_scan(
        self,
        post_process_scan,
        directory_access: DirectoryAccess,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that process-scan creates expected output files with correct URLs and file structure."""
        # Arrange
        base_url = f"{get_settings().base_url}/{RoutePrefix.EXTRACTOR}/files/{directory_access.token}"
        directory = get_settings().storage / f"{directory_access.tag}-{directory_access.token.hex}"

        # Act
        with monkeypatch.context() as mp:
            mp.setattr("preprocessors.router.create_vault", lambda _: directory_access)
            response = post_process_scan()

        # Assert
        expected_response = ProcessedDataAccess(
            scan=HttpUrl(f"{base_url}/scan.x3p"),
            preview=HttpUrl(f"{base_url}/preview.png"),
            surface_map=HttpUrl(f"{base_url}/surface_map.png"),
        )

        assert response.status_code == HTTPStatus.OK, "endpoint is alive"
        response_model = ProcessedDataAccess.model_validate(response.json())
        assert response_model == expected_response
        assert (directory / "scan.x3p").exists()
        assert (directory / "preview.png").exists()
        assert (directory / "surface_map.png").exists()

    def test_process_scan_overwrites_files(
        self,
        post_process_scan,
        directory_access: DirectoryAccess,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that processing the same scan file twice overwrites existing output files."""
        # Arrange
        monkeypatch.setattr("preprocessors.router.create_vault", lambda _: directory_access)
        directory = get_settings().storage / f"{directory_access.tag}-{directory_access.token.hex}"

        # Act I
        _ = post_process_scan()
        x3p_first_post_time = (directory / "scan.x3p").stat().st_mtime
        preview_first_post_time = (directory / "preview.png").stat().st_mtime

        # Assert
        assert x3p_first_post_time
        assert preview_first_post_time

        # Act II
        _ = post_process_scan()

        # Assert
        assert (directory / "scan.x3p").stat().st_mtime > x3p_first_post_time
        assert (directory / "preview.png").stat().st_mtime > preview_first_post_time

    @pytest.mark.parametrize(
        ("filename", "overhead"),
        [
            pytest.param("nonexistent.x3p", "None", id="nonexistent file"),
            pytest.param("unsuported.txt", "write_text is_unsupported_file", id="unsuported file"),
            pytest.param("empty.x3p", "touch", id="empty file"),
        ],
    )
    def test_process_scan_bad_file(self, filename: str, overhead: str, client: TestClient, tmp_path: Path) -> None:
        """Test that invalid scan files (nonexistent, unsupported, or empty) are rejected with 422 status."""
        # Arrange
        path = tmp_path / filename
        cmd, *args = overhead.strip().split()
        if func := getattr(path, cmd, None):
            func(*args)

        # Act - send raw JSON to bypass Pydantic model construction
        response = client.post(
            PROCESS_SCAN_ROUTE,
            json={"scan_file": str(path)},
        )

        # Assert - Pydantic validation should catch this
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        error_detail = response.json()["detail"]
        assert any("scan_file" in str(err) for err in error_detail)
