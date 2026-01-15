import os
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from models import DirectoryAccess


def test_pre_processors_placeholder(client: TestClient) -> None:
    """Test that the preprocessor root endpoint returns a placeholder message."""
    # Act
    response = client.get("/preprocessor")

    # Assert
    assert response.status_code == HTTPStatus.OK, "endpoint is alive"
    assert response.json() == {"message": "Hello from the pre-processors"}, "A placeholder response should be returned"


@pytest.mark.usefixtures("tmp_dir_api")
class TestPrepareMarkEndpoint:
    @pytest.fixture(autouse=True)
    def set_dir_to_test_env(self, monkeypatch: pytest.MonkeyPatch, directory_access: DirectoryAccess) -> None:
        """Set up the directory access for tests."""
        directory_access.resource_path.mkdir(exist_ok=True)
        monkeypatch.setattr("preprocessors.router.create_vault", lambda _: directory_access)

    def test_prepare_mark_endpoint_returns_urls(self, client: TestClient) -> None:
        """Test that the prepare-mark endpoint processes the request and returns file URLs."""
        # Arrange
        payload = {
            "project_name": "test_project",
            "scan_file": "packages/scratch-core/tests/resources/scans/circle.x3p",
            "mark_typ": "ejector_striation_mark",
            "mask_array": [[0, 1], [1, 0]],
            "rotation_angle": 15,
            "preprocessor_param": 1,
        }

        # Act
        response = client.post("/preprocessor/prepare-mark", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        json_response = response.json()
        expected_keys = [
            "scan",
            "preview",
            "surface_map",
            "mark_mat",
            "processed_mat",
            "profile_mat",
            "leveled_mat",
        ]
        for key in expected_keys:
            assert key in json_response, f"Response should contain URL for {key}"

    def test_prepare_mark_endpoint_has_made_files_in_vault(
        self, client: TestClient, directory_access: DirectoryAccess
    ) -> None:
        """Test that the prepare-mark endpoint creates files in the vault."""
        # Arrange
        payload = {
            "project_name": "test_project",
            "scan_file": "packages/scratch-core/tests/resources/scans/circle.x3p",
            "mark_typ": "ejector_striation_mark",
            "mask_array": [[0, 1], [1, 0]],
            "rotation_angle": 15,
            "preprocessor_param": 1,
        }
        # Act
        response = client.post("/preprocessor/prepare-mark", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        vault_path = directory_access.resource_path
        expected_files = [
            "scan.x3p",
            "preview.png",
            "surface_map.png",
            "mark.mat",
            "processed.mat",
            "profile.mat",
            "leveled.mat",
        ]
        pytest.xfail("Endpoint not implemented yet")  # TODO: Remove when endpoint is implemented
        for filename in expected_files:
            file_path = os.path.join(vault_path, filename)
            logger.info(f"Checking for file: {file_path}")
            assert os.path.exists(file_path), f"Expected file {filename} to be created in the vault"

    def test_prepare_mark_endpoint_response_url_matches_folder_location(
        self, client: TestClient, directory_access: DirectoryAccess
    ) -> None:
        """Test that the URLs in the prepare-mark endpoint response match the vault folder location."""
        # Arrange
        payload = {
            "project_name": "test_project",
            "scan_file": "packages/scratch-core/tests/resources/scans/circle.x3p",
            "mark_typ": "ejector_striation_mark",
            "mask_array": [[0, 1], [1, 0]],
            "rotation_angle": 15,
            "preprocessor_param": 1,
        }

        # Act
        response = client.post("/preprocessor/prepare-mark", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        json_response = response.json()
        base_url = f"http://localhost:8000/preprocessor/files/{payload['project_name']}/"
        for key, url in json_response.items():
            expected_url_start = base_url
            assert url.startswith(directory_access.access_url), f"URL for {key} should start with {expected_url_start}"
            # TODO: retrieve tag and token from url and find file in vault to ensure correctness
