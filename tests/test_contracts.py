import pytest
import requests
from pydantic import BaseModel
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND

ROOT_URL = "http://127.0.0.1:8000"


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
        ("get_request_url", "expected_response"),
        [
            ("/comparator", TemplateResponse),
            ("/pre-processor", TemplateResponse),
            ("/processor", TemplateResponse),
        ],
    )
    def test_root(self, get_request_url: str, expected_response: BaseModel) -> None:
        """Check if the root is still returning the hello world response."""
        # Act
        response = requests.get(f"{ROOT_URL}{get_request_url}", timeout=5)
        # Assert
        assert response.status_code == HTTP_200_OK
        expected_response.model_validate(response.json())

    def test_non_existing_contract(self) -> None:
        """Test if a non-existent contract returns 404."""
        # Act
        response = requests.get(f"{ROOT_URL}/non-existing-path", timeout=5)
        # Assert
        assert response.status_code == HTTP_404_NOT_FOUND
