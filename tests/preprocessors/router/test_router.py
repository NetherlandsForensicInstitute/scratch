import json
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pytest
from conversion.data_formats import MarkType
from conversion.leveling import SurfaceTerms
from fastapi.testclient import TestClient
from pydantic import HttpUrl
from scipy.constants import micro
from utils.constants import RegressionOrder

from constants import PreprocessorEndpoint, RoutePrefix
from extractors.schemas import GeneratedImages, PrepareMarkResponseImpression, PrepareMarkResponseStriation
from models import DirectoryAccess
from preprocessors.schemas import (
    EditImage,
    MaskParameters,
    PrepareMarkImpression,
    PrepareMarkStriation,
    PreprocessingImpressionParams,
    PreprocessingStriationParams,
)
from settings import get_settings


def test_pre_processors_placeholder(client: TestClient) -> None:
    """Test that the preprocessor root endpoint redirects to documentation."""
    # Act
    response = client.get(f"/{RoutePrefix.PREPROCESSOR}", follow_redirects=False)

    # Assert
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, "endpoint should redirect"
    assert response.headers["location"] == f"/docs#operations-tag-{RoutePrefix.PREPROCESSOR}", (
        "should redirect to preprocessor docs"
    )


@pytest.mark.parametrize(
    ("endpoint", "schema", "response_schema", "mark_parameters", "mark_type", "expected_keys"),
    [
        pytest.param(
            PreprocessorEndpoint.PREPARE_MARK_STRIATION,
            PrepareMarkStriation,
            PrepareMarkResponseStriation,
            PreprocessingStriationParams,
            MarkType.APERTURE_SHEAR_STRIATION,
            [
                "preview",
                "surface_map",
                "mark_data",
                "mark_meta",
                "processed_data",
                "processed_meta",
                "profile_data",
            ],
            id="striation mark",
        ),
        pytest.param(
            PreprocessorEndpoint.PREPARE_MARK_IMPRESSION,
            PrepareMarkImpression,
            PrepareMarkResponseImpression,
            PreprocessingImpressionParams,
            MarkType.CHAMBER_IMPRESSION,
            [
                "preview",
                "surface_map",
                "mark_data",
                "mark_meta",
                "processed_data",
                "processed_meta",
                "leveled_data",
                "leveled_meta",
            ],
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

    @pytest.fixture(autouse=True)
    def set_scan_file_path(self, scan_directory: Path):
        """Path to a dummy scan image file."""
        self.scan_file_path = scan_directory / "circle.x3p"

    def get_schema_for_endpoint(
        self,
        schema: type[PrepareMarkImpression | PrepareMarkStriation],
        mark_type: str,
        mask: list[list[float]],
        mark_parameters: type[PreprocessingStriationParams | PreprocessingImpressionParams],
    ):
        """Generate the schema payload for the prepare-mark endpoint."""
        return schema(
            project_name="test_project",
            mark_type=mark_type,  # type: ignore
            scan_file=self.scan_file_path,
            mask=mask,
            bounding_box_list=[[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]],
            mark_parameters=mark_parameters(),  # type: ignore
        ).model_dump(mode="json")

    def test_prepare_mark_endpoint_returns_urls(  # noqa: PLR0913
        self,
        client: TestClient,
        endpoint: PreprocessorEndpoint,
        schema: type[PrepareMarkImpression | PrepareMarkStriation],
        response_schema: type[PrepareMarkResponseImpression | PrepareMarkResponseStriation],
        mark_parameters: type[PreprocessingStriationParams | PreprocessingImpressionParams],
        mark_type: str,
        mask: list[list[float]],
        expected_keys: list[str],
    ) -> None:
        """Test that the prepare-mark endpoint processes the request and returns file URLs."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,
            mark_type=mark_type,
            mask=mask,
            mark_parameters=mark_parameters,
        )

        # Act
        response = client.post(f"/{RoutePrefix.PREPROCESSOR}/{endpoint}", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        json_response = response.json()

        for key in expected_keys:
            assert key in json_response, f"Response should contain URL for {key}"

    def test_prepare_mark_endpoint_has_made_files_in_vault(  # noqa: PLR0913
        self,
        client: TestClient,
        directory_access: DirectoryAccess,
        schema: type[PrepareMarkImpression | PrepareMarkStriation],
        response_schema: type[PrepareMarkResponseImpression | PrepareMarkResponseStriation],
        endpoint: PreprocessorEndpoint,
        mark_parameters: PreprocessingStriationParams | PreprocessingImpressionParams,
        mark_type: str,
        mask: list[list[float]],
        expected_keys: list[str],
    ) -> None:
        """Test that the prepare-mark endpoint creates files in the vault."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,  # type: ignore
            mark_type=mark_type,
            mask=mask,
            mark_parameters=mark_parameters,  # type: ignore
        )
        # Act
        response = client.post(f"/{RoutePrefix.PREPROCESSOR}/{endpoint}", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        expected_filenames = response_schema.get_files(directory_access.resource_path)

        missing = {path.name for path in expected_filenames.values() if not path.exists()}
        assert not missing, f"Expected: {', '.join(missing)} to be created"

    def test_prepare_mark_endpoint_response_url_matches_folder_location(  # noqa: PLR0913
        self,
        client: TestClient,
        directory_access: DirectoryAccess,
        schema: type[PrepareMarkImpression | PrepareMarkStriation],
        response_schema: type[PrepareMarkResponseImpression | PrepareMarkResponseStriation],
        endpoint: PreprocessorEndpoint,
        mark_parameters: PreprocessingStriationParams | PreprocessingImpressionParams,
        mark_type: str,
        mask: list[list[float]],
        expected_keys: list[str],
    ) -> None:
        """Test that the URLs in the prepare-mark endpoint response match the vault folder location."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,  # type: ignore
            mark_type=mark_type,
            mask=mask,
            mark_parameters=mark_parameters,  # type: ignore
        )

        # Act
        response = client.post(f"/{RoutePrefix.PREPROCESSOR}/{endpoint}", json=payload)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        json_response = response.json()
        base_url = f"http://localhost:8000/preprocessor/files/{payload['project_name']}/"
        for key, url in json_response.items():
            expected_url_start = base_url
            assert url.startswith(directory_access.access_url), f"URL for {key} should start with {expected_url_start}"
            # TODO: retrieve tag and token from url and find file in vault to ensure correctness


@pytest.mark.usefixtures("tmp_dir_api")
def test_edit_image_returns_valid_images(
    client: TestClient,
    directory_access: DirectoryAccess,
    scan_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests if the endpoint gives back the expected outcome."""
    # Arrange
    base_url = f"{get_settings().base_url}/{RoutePrefix.EXTRACTOR}/files/{directory_access.token}"
    directory = get_settings().storage / f"{directory_access.tag}-{directory_access.token.hex}"

    mask = np.array([[False] * 3, [False, True, False], [False] * 3], dtype=np.bool)

    params = EditImage(
        project_name="test",
        scan_file=scan_directory / "circle.x3p",
        mask_parameters=MaskParameters(shape=mask.shape),
        cutoff_length=2 * micro,
        resampling_factor=0.5,
        terms=SurfaceTerms.PLANE,
        regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        crop=True,
        step_size_x=1,
        step_size_y=1,
    )

    # Act
    with monkeypatch.context() as mp:
        mp.setattr("preprocessors.router.create_vault", lambda _: directory_access)
        response = client.post(
            f"{get_settings().base_url}/{RoutePrefix.PREPROCESSOR}/edit-scan",
            data={"params": json.dumps(params.model_dump(mode="json"))},
            files={"mask_data": ("mask.bin", mask.tobytes(order="C"), "application/octet-stream")},
            timeout=5,
        )

    # Assert
    expected_response = GeneratedImages(
        preview=HttpUrl(f"{base_url}/preview.png"),
        surface_map=HttpUrl(f"{base_url}/surface_map.png"),
    )
    assert response.status_code == HTTPStatus.OK, "endpoint is alive"
    response_model = GeneratedImages.model_validate(response.json())
    assert response_model == expected_response
    assert (directory / "preview.png").exists()
    assert (directory / "surface_map.png").exists()
