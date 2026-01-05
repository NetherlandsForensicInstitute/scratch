from http import HTTPStatus
from pathlib import Path

import pytest
from container_models.light_source import LightSource
from fastapi.testclient import TestClient
from PIL import Image
from starlette.status import HTTP_200_OK

from preprocessors import ProcessedDataLocation, UploadScan
from preprocessors.schemas import UploadScanParameters


def test_pre_processors_placeholder(client: TestClient) -> None:
    # Act
    response = client.get("/preprocessor")

    # Assert
    assert response.status_code == HTTP_200_OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the pre-processors"}, "A placeholder response should be returned"


@pytest.mark.e2e
class TestProcessScanEndpoint:
    """End-to-end tests for the /process-scan endpoint."""

    def test_process_scan_success_with_al3d_file(
        self, client: TestClient, scan_directory: Path, tmp_path: Path
    ) -> None:
        """Test successful scan processing with AL3D input file."""
        # Arrange
        scan_file = scan_directory / "circle.al3d"
        request_data = UploadScan(scan_file=scan_file, output_dir=tmp_path)

        # Act
        response = client.post(
            "/preprocessor/process-scan",
            json=request_data.model_dump(mode="json"),
        )

        # Assert - verify response
        assert response.status_code == HTTPStatus.OK
        result = ProcessedDataLocation.model_validate(response.json())

        # Assert - verify image files are valid PNGs
        with Image.open(result.preview_image) as img:
            assert img.format == "PNG"

        with Image.open(result.surfacemap_image) as img:
            assert img.format == "PNG"

    def test_process_scan_output_filenames_match_input(
        self, client: TestClient, scan_directory: Path, tmp_path: Path
    ) -> None:
        """Test that output filenames are derived from input filename."""
        # Arrange
        scan_file = scan_directory / "circle.al3d"
        request_data = UploadScan(scan_file=scan_file, output_dir=tmp_path)

        # Act
        response = client.post(
            "/preprocessor/process-scan",
            json=request_data.model_dump(mode="json"),
        )

        # Assert
        result = ProcessedDataLocation.model_validate(response.json())

        # Verify filenames contain the input stem "circle"
        assert result.x3p_image.name == "circle.x3p"
        assert result.preview_image.name == "circle_preview.png"
        assert result.surfacemap_image.name == "circle_surfacemap.png"

    def test_process_scan_with_custom_light_sources(
        self, client: TestClient, scan_directory: Path, tmp_path: Path
    ) -> None:
        """Test scan processing with custom light source configuration."""
        # Arrange
        scan_file = scan_directory / "circle.al3d"
        request_data = UploadScan(
            scan_file=scan_file,
            output_dir=tmp_path,
            parameters=UploadScanParameters(  # type: ignore
                light_sources=(
                    LightSource(azimuth=0, elevation=90),
                    LightSource(azimuth=90, elevation=45),
                    LightSource(azimuth=180, elevation=45),
                    LightSource(azimuth=270, elevation=45),
                ),
                observer=LightSource(azimuth=0, elevation=90),
            ),
        )

        # Act
        response = client.post(
            "/preprocessor/process-scan",
            json=request_data.model_dump(mode="json"),
        )

        # Assert
        assert response.status_code == HTTPStatus.OK
        result = ProcessedDataLocation.model_validate(response.json())

        # Verify surface map was generated with custom lighting
        assert result.surfacemap_image.exists()
        with Image.open(result.surfacemap_image) as img:
            assert img.format == "PNG"

    @pytest.mark.parametrize(
        ("filename", "side_effect"),
        [
            pytest.param("nonexistent.x3p", "None", id="nonexistent file"),
            pytest.param("unsuported.txt", "write_text is_unsupported_file", id="unsuported file"),
            pytest.param("empty.x3p", "touch", id="empty file"),
        ],
    )
    def test_process_scan_bad_file(
        self,
        filename: str,
        side_effect: str,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        """Test that Pydantic validation rejects nonexistent files."""
        # Arrange
        path = tmp_path / filename
        method, *args = side_effect.strip().split()
        if func := getattr(path, method, None):
            func(*args)

        # Act - send raw JSON to bypass Pydantic model construction
        response = client.post(
            "/preprocessor/process-scan",
            json={"scan_file": str(path), "output_dir": str(tmp_path)},
        )

        # Assert - Pydantic validation should catch this
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        error_detail = response.json()["detail"]
        assert any("scan_file" in str(err) for err in error_detail)
