from enum import StrEnum
from pathlib import Path

import pytest
import requests
from pydantic import BaseModel
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND

from constants import PROJECT_ROOT
from preprocessors import ProcessedDataLocation, UploadScan

ROOT_URL = "http://127.0.0.1:8000"
SCANS_DIR = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans"


class RoutePrefix(StrEnum):
    COMPARATOR = "comparator"
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

    @pytest.mark.parametrize(
        ("route", "expected_response"),
        (pytest.param(f"/{route}", TemplateResponse, id=route) for route in RoutePrefix),
    )
    def test_root(self, route: str, expected_response: BaseModel) -> None:
        """Check if the root is still returning the hello world response."""
        # Act
        response = requests.get(f"{ROOT_URL}{route}", timeout=5)
        # Assert
        assert response.status_code == HTTP_200_OK
        expected_response.model_validate(response.json())

    @pytest.fixture
    def process_scan(self, tmp_path: Path) -> tuple[BaseModel, type[BaseModel]]:
        """Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        data = UploadScan(scan_file=SCANS_DIR / "circle.x3p", output_dir=tmp_path)
        expected_response = ProcessedDataLocation
        # TODO: when implemented the file touch can ben removed.
        (tmp_path / "scan.x3p").touch()
        (tmp_path / "preview.png").touch()
        (tmp_path / "surface_map.png").touch()
        return data, expected_response

    @pytest.mark.parametrize(
        ("fixture_name", "sub_route"), [pytest.param("process_scan", "process-scan", id="process_scan")]
    )
    def test_pre_processor_post_requests(
        self, fixture_name: str, sub_route: str, request: pytest.FixtureRequest
    ) -> None:
        """Test if the process scan endpoint returns an expected model."""
        data, expected_response = request.getfixturevalue(fixture_name)
        # Act
        response = requests.post(
            f"{ROOT_URL}/{RoutePrefix.PREPROCESSOR}/{sub_route}", json=data.model_dump(mode="json"), timeout=5
        )
        # Assert
        assert response.status_code == HTTP_200_OK
        expected_response.model_validate(response.json())

    def test_non_existing_contract(self) -> None:
        """Test if a non-existent contract returns 404."""
        # Act
        response = requests.get(f"{ROOT_URL}/non-existing-path", timeout=5)
        # Assert
        assert response.status_code == HTTP_404_NOT_FOUND
