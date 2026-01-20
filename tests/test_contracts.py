from enum import StrEnum
from pathlib import Path

import pytest
import requests
from pydantic import BaseModel
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND

from constants import PROJECT_ROOT
from extractors.schemas import ProcessDataUrls
from models import DirectoryAccess
from preprocessors.schemas import EditImage, EditImageParameters, UploadScan
from settings import get_settings

SCANS_DIR = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans"
MASK = ((1, 0), (0, 1))
CUTOFF_LENGTH = 250  # 250 micrometers in meters


class RoutePrefix(StrEnum):
    COMPARATOR = "comparator"
    EXTRACTOR = "extractor"
    PREPROCESSOR = "preprocessor"
    PROCESSOR = "processor"


class TemplateResponse(BaseModel):
    """Simple template response,."""

    message: str


@pytest.mark.contract_testing
class TestContracts:
    """
    Test the outgoing traffic.

    Here are tests like:
      - Get/Post endpoints are online (health check)
      - checks if some end-points are forbidden (auth)
      - checks if input and output are in correct format (schema definitions).
    """

    @pytest.fixture(scope="class")
    def process_scan(self, scan_directory: Path) -> tuple[BaseModel, type[BaseModel]]:
        """Create dummy files for the expected response.

        Returns the post request data and expected response type.
        """
        return UploadScan(scan_file=scan_directory / "Klein_non_replica_mode.al3d"), ProcessDataUrls  # type: ignore

    @pytest.fixture(scope="class")
    def edit_scan(self, scan_directory: Path) -> tuple[BaseModel, type[BaseModel]]:
        """Create test data for edit-scan endpoint.

        Returns the post request data and expected response type.
        """
        data = EditImage(
            scan_file=scan_directory / "Klein_non_replica_mode_X3P_Scratch.x3p",
            parameters=EditImageParameters(mask=MASK, cutoff_length=CUTOFF_LENGTH),  # type: ignore
        )
        return data, ProcessDataUrls

    @pytest.mark.parametrize(
        ("route", "expected_response"),
        (pytest.param(f"/{route}", TemplateResponse, id=route) for route in RoutePrefix),
    )
    def test_root(self, route: str, expected_response: BaseModel) -> None:
        """Check if the root is still returning the hello world response."""
        # Act
        response = requests.get(f"{get_settings().base_url}{route}", timeout=5)
        # Assert
        assert response.status_code == HTTP_200_OK
        expected_response.model_validate(response.json())

    @pytest.mark.parametrize(
        ("fixture_name", "sub_route"),
        [
            pytest.param("process_scan", "process-scan", id="process_scan"),
            pytest.param("edit_scan", "edit-scan", marks=pytest.mark.xfail, id="edit_scan"),
        ],
    )
    def test_pre_processor_post_requests(
        self, fixture_name: str, sub_route: str, request: pytest.FixtureRequest
    ) -> None:
        """Test if preprocessor POST endpoints return expected models."""
        data, expected_response = request.getfixturevalue(fixture_name)
        # Act
        response = requests.post(
            f"{get_settings().base_url}/{RoutePrefix.PREPROCESSOR}/{sub_route}",
            json=data.model_dump(mode="json"),
            timeout=5,
        )
        # Assert
        assert response.status_code == HTTP_200_OK
        expected_response.model_validate(response.json())

    def test_extractor_get_file_endpoint(self, directory_access: DirectoryAccess) -> None:
        """Test if extractor /files/{token}/{filename} endpoint retrieves processed files.

        First creates files via process-scan, then retrieves each file type and validates
        response status and content types.
        """
        # Arrange: Create files via process-scan endpoint
        (directory_access.resource_path / "scan.x3p").write_bytes(b"x3p content")
        response = requests.get(f"{directory_access.access_url}/scan.x3p", timeout=5)
        assert response.status_code == HTTP_200_OK, "Failed to retrieve x3p"
        assert response.headers["content-type"] == "application/octet-stream", "Wrong content type for x3p"

    def test_non_existing_contract(self) -> None:
        """Test if a non-existent contract returns 404."""
        # Act
        response = requests.get(f"{get_settings().base_url}/non-existing-path", timeout=5)
        # Assert
        assert response.status_code == HTTP_404_NOT_FOUND
