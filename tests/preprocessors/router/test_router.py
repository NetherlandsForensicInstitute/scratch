import os
from http import HTTPStatus
from pathlib import Path

import pytest
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from fastapi.testclient import TestClient
from loguru import logger

from constants import ImpressionMarks, MaskTypes, StriationMarks
from models import DirectoryAccess
from preprocessors.schemas import (
    BaseParameters,
    CropInfo,
    PrepareMarkImpression,
    PrepareMarkStriation,
    PreprocessingStriationParams,
)


def test_pre_processors_placeholder(client: TestClient) -> None:
    """Test that the preprocessor root endpoint returns a placeholder message."""
    # Act
    response = client.get("/preprocessor")

    # Assert
    assert response.status_code == HTTPStatus.OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the pre-processors"}, "A placeholder response should be returned"


@pytest.mark.parametrize(
    ("subroute", "schema", "mark_parameters", "mark_type", "expected_files"),
    [
        pytest.param(
            "/prepare-mark-striation",
            PrepareMarkStriation,
            PreprocessingStriationParams,
            StriationMarks.APERTURE_SHEAR,
            ["scan", "preview", "surface_map", "mark_file", "processed_file", "profile_file"],
            id="striation mark",
        ),
        pytest.param(
            "/prepare-mark-impression",
            PrepareMarkImpression,
            PreprocessingImpressionParams,
            ImpressionMarks.CHAMBER,
            ["scan", "preview", "surface_map", "mark_file", "processed_file", "leveled_file"],
            id="impression mark",
        ),
    ],
)
@pytest.mark.usefixtures("tmp_dir_api")
class TestPrepareMarkEndpoint:
    @pytest.fixture(autouse=True)
    def set_dir_to_test_env(self, monkeypatch: pytest.MonkeyPatch, directory_access: DirectoryAccess) -> None:
        """Set up the directory access for tests."""
        directory_access.resource_path.mkdir(exist_ok=True)
        monkeypatch.setattr("preprocessors.router.create_vault", lambda _: directory_access)

    def get_schema_for_endpoint(
        self,
        schema: type[PrepareMarkImpression | PrepareMarkStriation],
        mark_type: str,
        mark_parameters: type[PreprocessingStriationParams | PreprocessingImpressionParams],
    ):
        """Generate the schema payload for the prepare-mark endpoint."""
        return schema(
            project_name="test_project",
            mark_type=mark_type,  # type: ignore
            scan_file=Path("packages/scratch-core/tests/resources/scans/circle.x3p"),
            mask_array=[[0, 1], [1, 0]],
            crop_info=CropInfo(type=MaskTypes.CIRCLE, data={}, is_foreground=False),
            rotation_angle=15,
            mark_parameters=mark_parameters(),  # type: ignore
        ).model_dump(mode="json")

    def test_prepare_mark_endpoint_returns_urls(  # noqa: PLR0913
        self,
        client: TestClient,
        subroute: str,
        schema: type[PrepareMarkImpression | PrepareMarkStriation],
        mark_parameters: type[PreprocessingStriationParams | PreprocessingImpressionParams],
        mark_type: str,
        expected_files: list[str],
    ) -> None:
        """Test that the prepare-mark endpoint processes the request and returns file URLs."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,
            mark_type=mark_type,
            mark_parameters=mark_parameters,
        )

        # Act
        response = client.post(f"/preprocessor{subroute}", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        json_response = response.json()

        for key in expected_files:
            assert key in json_response, f"Response should contain URL for {key}"

    def test_prepare_mark_endpoint_has_made_files_in_vault(  # noqa: PLR0913
        self,
        client: TestClient,
        directory_access: DirectoryAccess,
        schema: type[BaseParameters],
        subroute: str,
        mark_parameters: PreprocessingStriationParams | PreprocessingImpressionParams,
        mark_type: str,
        expected_files: list[str],
    ) -> None:
        """Test that the prepare-mark endpoint creates files in the vault."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,  # type: ignore
            mark_type=mark_type,
            mark_parameters=mark_parameters,  # type: ignore
        )
        # Act
        response = client.post(f"/preprocessor{subroute}", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        vault_path = directory_access.resource_path
        pytest.xfail("Endpoint not implemented yet")  # TODO: Remove when endpoint is implemented
        for filename in expected_files:
            file_path = os.path.join(vault_path, filename)
            logger.info(f"Checking for file: {file_path}")
            assert os.path.exists(file_path), f"Expected file {filename} to be created in the vault"

    def test_prepare_mark_endpoint_response_url_matches_folder_location(  # noqa: PLR0913
        self,
        client: TestClient,
        directory_access: DirectoryAccess,
        schema: type[BaseParameters],
        subroute: str,
        mark_parameters: PreprocessingStriationParams | PreprocessingImpressionParams,
        mark_type: str,
        expected_files: list[str],
    ) -> None:
        """Test that the URLs in the prepare-mark endpoint response match the vault folder location."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,  # type: ignore
            mark_type=mark_type,
            mark_parameters=mark_parameters,  # type: ignore
        )

        # Act
        response = client.post(f"/preprocessor{subroute}", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        json_response = response.json()
        base_url = f"http://localhost:8000/preprocessor/files/{payload['project_name']}/"
        for key, url in json_response.items():
            expected_url_start = base_url
            assert url.startswith(directory_access.access_url), f"URL for {key} should start with {expected_url_start}"
            # TODO: retrieve tag and token from url and find file in vault to ensure correctness
