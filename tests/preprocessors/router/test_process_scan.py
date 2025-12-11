from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from container_models.light_source import LightSource
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import HttpUrl

from main import app
from preprocessors import ProcessedDataLocation, UploadScan
from preprocessors.router import ROUTE
from preprocessors.schemas import UploadScanParameters


@pytest.fixture(scope="module")
def upload_scan(scan_directory: Path) -> UploadScan:
    """Fixture that provides a default UploadScan model using circle.al3d."""
    return UploadScan(scan_file=scan_directory / "Klein_non_replica_mode.al3d")


@pytest.fixture
def post_process_scan(client: TestClient, upload_scan: UploadScan):
    """Fixture that provides a function to post to the process-scan endpoint.

    Uses upload_scan (AL3D) by default, but can accept a custom UploadScan model.
    """

    def _post(input_model: UploadScan | None = None):
        model = input_model if input_model is not None else upload_scan
        return client.post(f"{ROUTE}/process-scan", json=model.model_dump(mode="json"))

    return _post


@pytest.mark.e2e
class TestProcessScanEndpoint:
    """End-to-end tests for the /process-scan endpoint."""

    def test_process_scan_success_with_al3d_file(self, post_process_scan, client: TestClient) -> None:
        """Test successful scan processing with AL3D input file."""
        # Act I
        response = post_process_scan()

        # Assert - verify response
        assert response.status_code == HTTPStatus.OK
        result = ProcessedDataLocation.model_validate(response.json())
        assert result.preview_image.path
        assert result.surfacemap_image.path
        assert result.x3p_image.path

        # Act II
        preview = client.get(result.preview_image.path)
        surfacemap = client.get(result.surfacemap_image.path)
        x3p_response = client.get(result.x3p_image.path)

        # Assert - verify response status codes
        assert preview.status_code == HTTPStatus.OK
        assert surfacemap.status_code == HTTPStatus.OK
        assert x3p_response.status_code == HTTPStatus.OK

        # Assert - verify content types
        assert preview.headers["content-type"] == "image/png"
        assert surfacemap.headers["content-type"] == "image/png"
        assert x3p_response.headers["content-type"] == "application/octet-stream"

        # Assert - verify PNG files are valid images
        assert Image.open(BytesIO(preview.content))
        assert Image.open(BytesIO(surfacemap.content))

        # Assert - verify x3p file has content
        assert len(x3p_response.content) > 0

    def test_process_scan_with_custom_light_sources(
        self, post_process_scan, upload_scan: UploadScan, client: TestClient
    ) -> None:
        """Test that custom light sources produce different surface maps with varying brightness."""
        # Arrange
        single_light = LightSource(azimuth=90, elevation=45)
        observer = LightSource(azimuth=0, elevation=90)

        # Control: one light source
        control_data = UploadScan(
            scan_file=upload_scan.scan_file,
            parameters=UploadScanParameters(  # type: ignore
                light_sources=(single_light,),
                observer=observer,
            ),
        )

        # Result: same light source doubled
        request_data = UploadScan(
            scan_file=upload_scan.scan_file,
            parameters=UploadScanParameters(  # type: ignore
                light_sources=(single_light, LightSource(azimuth=180, elevation=45)),
                observer=observer,
            ),
        )

        # Act I
        control_response = post_process_scan(control_data)
        response = post_process_scan(request_data)

        # Assert
        assert control_response.status_code == HTTPStatus.OK
        assert response.status_code == HTTPStatus.OK
        result_location = ProcessedDataLocation.model_validate(response.json())
        control_location = ProcessedDataLocation.model_validate(control_response.json())
        assert result_location.surfacemap_image.path
        assert control_location.surfacemap_image.path

        # Act II
        control = client.get(control_location.surfacemap_image.path)
        result = client.get(result_location.surfacemap_image.path)

        # Assert - verify responses are successful
        assert control.status_code == HTTPStatus.OK
        assert result.status_code == HTTPStatus.OK

        # Assert - verify that the two surfacemaps are not the same
        assert control.content != result.content, "Surfacemaps should differ with different light sources"

        # Assert - verify that result with doubled light sources is brighter
        control_img = Image.open(BytesIO(control.content))
        result_img = Image.open(BytesIO(result.content))

        # Calculate average brightness for each image
        control_brightness = np.array(control_img).mean()
        result_brightness = np.array(result_img).mean()

        assert result_brightness > control_brightness, (
            f"Result brightness ({result_brightness:.2f}) with two light sources should be greater than "
            f"control brightness ({control_brightness:.2f}) with one light source"
        )


@pytest.mark.integration
class TestProcessScan:
    """Integration tests for the /process-scan endpoint with file system operations."""

    def test_process_scan(self, post_process_scan, tmp_dir_api: Path, token: UUID) -> None:
        """Test that process-scan creates expected output files with correct URLs and file structure."""
        # Arrange
        tokenized_base_name = f"{token}/Klein_non_replica_mode"
        base_url = f"http://localhost:8000{ROUTE}/file/{tokenized_base_name}"

        # Act
        response = post_process_scan()

        # Assert
        expected_response = ProcessedDataLocation(
            x3p_image=HttpUrl(f"{base_url}.x3p"),
            preview_image=HttpUrl(f"{base_url}_preview.png"),
            surfacemap_image=HttpUrl(f"{base_url}_surfacemap.png"),
        )

        assert response.status_code == HTTPStatus.OK, "endpoint is alive"
        response_model = ProcessedDataLocation.model_validate(response.json())
        assert response_model == expected_response
        assert (tmp_dir_api / f"{tokenized_base_name}.x3p").exists()
        assert (tmp_dir_api / f"{tokenized_base_name}_preview.png").exists()
        assert (tmp_dir_api / f"{tokenized_base_name}_surfacemap.png").exists()

    def test_process_scan_overwrites_files(self, post_process_scan, tmp_dir_api: Path, token: UUID) -> None:
        """Test that processing the same scan file twice overwrites existing output files."""
        # Arrange
        tokenized_base_name = f"{token}/Klein_non_replica_mode"

        # Act
        _ = post_process_scan()
        x3p_first_post_time = (tmp_dir_api / f"{tokenized_base_name}.x3p").stat().st_mtime
        preview_first_post_time = (tmp_dir_api / f"{tokenized_base_name}_preview.png").stat().st_mtime
        _ = post_process_scan()

        # Assert
        assert (tmp_dir_api / f"{tokenized_base_name}.x3p").stat().st_mtime > x3p_first_post_time
        assert (tmp_dir_api / f"{tokenized_base_name}_preview.png").stat().st_mtime > preview_first_post_time

    def test_process_scan_files_are_deleted_after_restart(self, upload_scan: UploadScan, token: UUID) -> None:
        """Test that temporary files are automatically cleaned up when the application shuts down."""
        # Arrange
        preview_path = f"{token}/Klein_non_replica_mode_preview.png"
        # Act
        with TestClient(app) as client:
            client.post(f"{ROUTE}/process-scan", json=upload_scan.model_dump(mode="json"))
            # Assert
            tmp_dir = Path(str(app.state.temp_dir.name))
            assert (tmp_dir / preview_path).exists()
        assert not (tmp_dir / preview_path).exists(), "Temp dir should be removed after app shutdown"

    @pytest.mark.parametrize(
        ("filename", "side_effect"),
        [
            pytest.param("nonexistent.x3p", "None", id="nonexistent file"),
            pytest.param("unsuported.txt", "write_text is_unsupported_file", id="unsuported file"),
            pytest.param("empty.x3p", "touch", id="empty file"),
        ],
    )
    def test_process_scan_bad_file(self, filename: str, side_effect: str, client: TestClient, tmp_path: Path) -> None:
        """Test that invalid scan files (nonexistent, unsupported, or empty) are rejected with 422 status."""
        # Arrange
        path = tmp_path / filename
        method, *args = side_effect.strip().split()
        if func := getattr(path, method, None):
            func(*args)

        # Act - send raw JSON to bypass Pydantic model construction
        response = client.post(
            f"{ROUTE}/process-scan",
            json={"scan_file": str(path)},
        )

        # Assert - Pydantic validation should catch this
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        error_detail = response.json()["detail"]
        assert any("scan_file" in str(err) for err in error_detail)
