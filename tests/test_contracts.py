from enum import StrEnum
from http import HTTPStatus
from pathlib import Path

import pytest
import requests
from pydantic import BaseModel

from constants import PROJECT_ROOT
from extractors.schemas import (
    ComparisonResponseImpression,
    ComparisonResponseStriation,
    GeneratedImages,
    PrepareMarkResponseImpression,
    PrepareMarkResponseStriation,
    ProcessedDataAccess,
)
from models import DirectoryAccess
from preprocessors.schemas import (
    EditImage,
    PrepareMarkImpression,
    PrepareMarkStriation,
    PreprocessingImpressionParams,
    PreprocessingStriationParams,
    UploadScan,
)
from processors.schemas import (
    CalculateScoreImpression,
    CalculateScoreStriation,
    ImpressionParameters,
    StriationParamaters,
)
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


type Interface = tuple[BaseModel, type[BaseModel]]


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
    def process_scan(self, scan_directory: Path) -> Interface:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        return UploadScan(scan_file=scan_directory / "circle.x3p"), ProcessedDataAccess  # type: ignore

    @pytest.fixture(scope="class")
    def prepare_mark_impression(self, scan_directory: Path, mask: list[list[float]]) -> Interface:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        return PrepareMarkImpression(
            scan_file=scan_directory / "circle.x3p",
            mark_type="breech face impression mark",
            mask=mask,
            bounding_box_list=[[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]],
            mark_parameters=PreprocessingImpressionParams(),
        ), PrepareMarkResponseImpression  # type: ignore

    @pytest.fixture(scope="class")
    def prepare_mark_striation(self, scan_directory: Path, mask: list[list[float]]) -> Interface:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        return PrepareMarkStriation(
            scan_file=scan_directory / "circle.x3p",
            mark_type="aperture shear striation mark",
            mask=mask,
            bounding_box_list=[[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]],
            mark_parameters=PreprocessingStriationParams(),
        ), PrepareMarkResponseStriation  # type: ignore

    @pytest.fixture(scope="class")
    def edit_scan(self, scan_directory: Path) -> Interface:
        """Create test data for edit-scan endpoint.

        Returns the post request data and expected response type.
        """
        data = EditImage(  # type: ignore
            scan_file=scan_directory / "Klein_non_replica_mode_X3P_Scratch.x3p",
            mask=MASK,
            cutoff_length=CUTOFF_LENGTH,
        )
        return data, GeneratedImages

    @pytest.fixture(scope="class")
    def calculate_score_impression(self, directory_access: DirectoryAccess) -> Interface:
        """
        Create test data for calculate-score-impression endpoint.

        Returns the post request data and expected response type.
        """
        return CalculateScoreImpression(
            mark_dir_ref=directory_access.resource_path,
            mark_dir_comp=directory_access.resource_path,
            param=ImpressionParameters(),
        ), ComparisonResponseImpression

    @pytest.fixture(scope="class")
    def calculate_score_striation(self, directory_access: DirectoryAccess) -> Interface:
        """
        Create test data for calculate-score-striation endpoint.

        Returns the post request data and expected response type.
        """
        return CalculateScoreStriation(
            mark_dir_ref=directory_access.resource_path,
            mark_dir_comp=directory_access.resource_path,
            param=StriationParamaters(),
        ), ComparisonResponseStriation

    @pytest.mark.parametrize(
        "route",
        (pytest.param(f"/{route}", id=route) for route in RoutePrefix),
    )
    def test_root(self, route: str) -> None:
        """Check if the root redirects to the documentation section."""
        # Act
        response = requests.get(f"{get_settings().base_url}{route}", timeout=5, allow_redirects=False)
        # Assert
        assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, (
            "endpoint should redirect with temporary redirect status"
        )
        expected_location = f"/docs#operations-tag-{route.lstrip('/')}"
        assert response.headers["location"] == expected_location, f"should redirect to {expected_location}"

    @pytest.mark.parametrize(
        ("fixture_name", "sub_route"),
        [
            pytest.param("process_scan", "process-scan", id="process_scan"),
            pytest.param("prepare_mark_impression", "prepare-mark-impression", id="prepare_mark_impression"),
            pytest.param("prepare_mark_striation", "prepare-mark-striation", id="prepare_mark_striation"),
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
        assert response.status_code == HTTPStatus.OK
        expected_response.model_validate(response.json())

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        ("fixture_name", "sub_route"),
        [
            pytest.param(
                "calculate_score_impression",
                "calculate-score-impression",
                id="calculate_score_impression",
            ),
            pytest.param(
                "calculate_score_striation",
                "calculate-score-striation",
                id="calculate_score_striation",
            ),
        ],
    )
    def test_processor_post_requests(self, fixture_name: str, sub_route: str, request: pytest.FixtureRequest) -> None:
        """Test if processor POST endpoints return expected models."""
        data, expected_response = request.getfixturevalue(fixture_name)
        # Act
        response = requests.post(
            f"{get_settings().base_url}/{RoutePrefix.PROCESSOR}/{sub_route}",
            json=data.model_dump(mode="json"),
            timeout=5,
        )
        # Assert
        assert response.status_code == HTTPStatus.OK
        expected_response.model_validate(response.json())

    def test_extractor_get_file_endpoint(self, directory_access: DirectoryAccess) -> None:
        """
        Test if extractor /files/{token}/{filename} endpoint retrieves processed files.

        First creates files via process-scan, then retrieves each file type and validates
        response status and content types.
        """
        # Arrange: Create files via process-scan endpoint
        (directory_access.resource_path / "scan.x3p").write_bytes(b"x3p content")
        response = requests.get(f"{directory_access.access_url}/scan.x3p", timeout=5)
        assert response.status_code == HTTPStatus.OK, "Failed to retrieve x3p"
        assert response.headers["content-type"] == "application/octet-stream", "Wrong content type for x3p"

    def test_non_existing_contract(self) -> None:
        """Test if a non-existent contract returns 404."""
        # Act
        response = requests.get(f"{get_settings().base_url}/non-existing-path", timeout=5)
        # Assert
        assert response.status_code == HTTPStatus.NOT_FOUND
