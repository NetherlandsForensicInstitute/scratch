from enum import StrEnum

import pytest
import requests
from pydantic import BaseModel
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND

from constants import PROJECT_ROOT
from pre_processors.schemas import ProcessScan, UploadScan

ROOT_URL = "http://127.0.0.1:8000"
SCANS_DIR = PROJECT_ROOT / "packages/scratch-core/tests/resources/scans"


class RootRout(StrEnum):
    COMPARATOR = "comparator"
    PRE_PROCESSOR = "pre-processor"
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
        ("rout", "expected_response"),
        (pytest.param(f"/{rout}", TemplateResponse, id=rout) for rout in RootRout),
    )
    def test_root(self, rout: str, expected_response: BaseModel) -> None:
        """Check if the root is still returning the hello world response."""
        # Act
        response = requests.get(f"{ROOT_URL}{rout}", timeout=5)
        # Assert
        assert response.status_code == HTTP_200_OK
        expected_response.model_validate(response.json())

    @pytest.mark.parametrize(
        ("sub_rout", "data", "expected_response"),
        [
            pytest.param(
                "/process-scan",
                UploadScan(scan_file=SCANS_DIR / "circle.x3p", output_dir=SCANS_DIR),
                ProcessScan,
                id="process-scan",
            ),
        ],
    )
    def test_pre_processor_post_requests(self, sub_rout: str, data: BaseModel, expected_response: BaseModel) -> None:
        """Test if the process scan endpoint returns an expected model."""
        # Act
        response = requests.post(f"{ROOT_URL}/{RootRout.PRE_PROCESSOR}{sub_rout}", data.model_dump_json(), timeout=5)
        # Assert
        assert response.status_code == HTTP_200_OK
        expected_response.model_validate(response.json())

    def test_non_existing_contract(self) -> None:
        """Test if a non-existent contract returns 404."""
        # Act
        response = requests.get(f"{ROOT_URL}/non-existing-path", timeout=5)
        # Assert
        assert response.status_code == HTTP_404_NOT_FOUND
