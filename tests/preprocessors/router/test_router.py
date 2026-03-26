import json
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pytest
from container_models.base import BinaryMask
from conversion.data_formats import MarkType
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import HttpUrl
from scipy.constants import micro
from utils.constants import RegressionOrder

from constants import PreprocessorEndpoint, RoutePrefix
from models import DirectoryAccess
from preprocessors.constants import PrepareMarkImpressionFiles, PrepareMarkStriationFiles, SurfaceOptions
from preprocessors.schemas import (
    EditImage,
    GeneratedImages,
    PrepareMarkImpression,
    PrepareMarkResponseImpression,
    PrepareMarkResponseStriation,
    PrepareMarkStriation,
    PreprocessingImpressionParams,
    PreprocessingStriationParams,
)
from settings import get_settings


def send_post_request_with_mask(client: TestClient, endpoint: str, params: dict, mask: BinaryMask) -> Response:
    return client.post(
        f"{get_settings().base_url}/{RoutePrefix.PREPROCESSOR}/{endpoint}",
        data={"params": json.dumps(params, default=str)},
        files={"mask_data": ("mask.bin", mask.tobytes(order="C"), "application/octet-stream")},
        timeout=5,
    )


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
    ("endpoint", "schema", "response_schema", "mark_parameters", "mark_type", "expected_keys", "files"),
    [
        pytest.param(
            PreprocessorEndpoint.PREPARE_MARK_STRIATION,
            PrepareMarkStriation,
            PrepareMarkResponseStriation,
            PreprocessingStriationParams,
            MarkType.APERTURE_SHEAR_STRIATION,
            [
                "preview_image",
                "surface_map_image",
                "mark_data",
                "mark_meta",
                "processed_data",
                "processed_meta",
                "profile_data",
            ],
            PrepareMarkStriationFiles,
            id="striation mark",
        ),
        pytest.param(
            PreprocessorEndpoint.PREPARE_MARK_IMPRESSION,
            PrepareMarkImpression,
            PrepareMarkResponseImpression,
            PreprocessingImpressionParams,
            MarkType.CHAMBER_IMPRESSION,
            [
                "preview_image",
                "surface_map_image",
                "mark_data",
                "mark_meta",
                "processed_data",
                "processed_meta",
                "leveled_data",
                "leveled_meta",
            ],
            PrepareMarkImpressionFiles,
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
        mark_parameters: type[PreprocessingStriationParams | PreprocessingImpressionParams],
    ):
        """Generate the schema payload for the prepare-mark endpoint."""
        return schema(
            project_name="test_project",
            mark_type=mark_type,  # type: ignore
            scan_file=self.scan_file_path,
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
        mask: BinaryMask,
        expected_keys: list[str],
        files: type[PrepareMarkImpressionFiles | PrepareMarkStriationFiles],
    ) -> None:
        """Test that the prepare-mark endpoint processes the request and returns file URLs."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,
            mark_type=mark_type,
            mark_parameters=mark_parameters,
        )

        # Act
        response = send_post_request_with_mask(client=client, endpoint=endpoint, params=payload, mask=mask)

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
        mask: BinaryMask,
        mark_type: str,
        expected_keys: list[str],
        files: type[PrepareMarkImpressionFiles | PrepareMarkStriationFiles],
    ) -> None:
        """Test that the prepare-mark endpoint creates files in the vault."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,  # type: ignore
            mark_type=mark_type,
            mark_parameters=mark_parameters,  # type: ignore
        )
        # Act
        response = send_post_request_with_mask(client=client, endpoint=endpoint, params=payload, mask=mask)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        expected_absolute_file_paths = [file.get_file_path(directory_access.resource_path) for file in files]
        missing = {path.name for path in expected_absolute_file_paths if not path.exists()}
        assert not missing, f"Expected: {', '.join(missing)} to be created"

    def test_prepare_mark_endpoint_response_url_matches_folder_location(  # noqa: PLR0913
        self,
        client: TestClient,
        directory_access: DirectoryAccess,
        schema: type[PrepareMarkImpression | PrepareMarkStriation],
        response_schema: type[PrepareMarkResponseImpression | PrepareMarkResponseStriation],
        endpoint: PreprocessorEndpoint,
        mark_parameters: PreprocessingStriationParams | PreprocessingImpressionParams,
        mask: BinaryMask,
        mark_type: str,
        expected_keys: list[str],
        files: type[PrepareMarkImpressionFiles | PrepareMarkStriationFiles],
    ) -> None:
        """Test that the URLs in the prepare-mark endpoint response match the vault folder location."""
        # Arrange
        payload = self.get_schema_for_endpoint(
            schema=schema,  # type: ignore
            mark_type=mark_type,
            mark_parameters=mark_parameters,  # type: ignore
        )

        # Act
        response = send_post_request_with_mask(client=client, endpoint=endpoint, params=payload, mask=mask)

        # Assert
        assert response.status_code == HTTPStatus.OK, f"endpoint is alive, {response.text}"
        json_response = response.json()
        base_url = f"http://localhost:8000/preprocessor/files/{payload['project_name']}/"
        for key, url in json_response.items():
            expected_url_start = base_url
            assert url.startswith(directory_access.access_url), f"URL for {key} should start with {expected_url_start}"
            # TODO: retrieve tag and token from url and find file in vault to ensure correctness


_MARK_ENDPOINT_CASES = [
    pytest.param(
        PreprocessorEndpoint.PREPARE_MARK_STRIATION,
        PrepareMarkStriation,
        PreprocessingStriationParams,
        MarkType.APERTURE_SHEAR_STRIATION,
        "preprocessors.router.process_prepare_striation_mark",
        id="striation mark",
    ),
    pytest.param(
        PreprocessorEndpoint.PREPARE_MARK_IMPRESSION,
        PrepareMarkImpression,
        PreprocessingImpressionParams,
        MarkType.CHAMBER_IMPRESSION,
        "preprocessors.router.process_prepare_impression_mark",
        id="impression mark",
    ),
]


@pytest.mark.parametrize(
    ("endpoint", "schema", "mark_parameters", "mark_type"),
    [
        pytest.param(
            PreprocessorEndpoint.PREPARE_MARK_STRIATION,
            PrepareMarkStriation,
            PreprocessingStriationParams,
            MarkType.APERTURE_SHEAR_STRIATION,
            id="striation mark",
        ),
        pytest.param(
            PreprocessorEndpoint.PREPARE_MARK_IMPRESSION,
            PrepareMarkImpression,
            PreprocessingImpressionParams,
            MarkType.CHAMBER_IMPRESSION,
            id="impression mark",
        ),
    ],
)
def test_prepare_mark_returns_422_on_mask_shape_mismatch(  # noqa: PLR0913
    client: TestClient,
    directory_access: DirectoryAccess,
    scan_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: PreprocessorEndpoint,
    schema: type[PrepareMarkImpression | PrepareMarkStriation],
    mark_parameters: type[PreprocessingStriationParams | PreprocessingImpressionParams],
    mark_type: MarkType,
) -> None:
    """Test that a 422 is returned when the mask shape does not match the scan image shape."""
    wrong_mask = np.zeros(shape=(2, 2), dtype=np.bool_)  # 2x2, won't match the scan shape

    payload = schema(
        project_name="test_project",
        mark_type=mark_type,
        scan_file=scan_directory / "circle.x3p",
        mark_parameters=mark_parameters(),  # type: ignore
        bounding_box_list=[],
    ).model_dump(mode="json")

    with monkeypatch.context() as mp:
        mp.setattr("preprocessors.router.create_vault", lambda _: directory_access)
        response = send_post_request_with_mask(client=client, endpoint=endpoint, params=payload, mask=wrong_mask)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.parametrize(
    ("endpoint", "schema", "mark_parameters", "mark_type", "controller_function"),
    _MARK_ENDPOINT_CASES,
)
def test_prepare_mark_returns_422_on_scan_parse_error(
    client: TestClient,
    directory_access: DirectoryAccess,
    scan_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask: BinaryMask,
    endpoint: PreprocessorEndpoint,
    schema: type[PrepareMarkImpression | PrepareMarkStriation],
    mark_parameters: type[PreprocessingStriationParams | PreprocessingImpressionParams],
    mark_type: MarkType,
    controller_function: str,
) -> None:
    """Test that a 422 is returned when the scan file cannot be parsed."""
    payload = schema(
        project_name="test_project",
        mark_type=mark_type,
        scan_file=scan_directory / "circle.x3p",
        mark_parameters=mark_parameters(),  # type: ignore
        bounding_box_list=[],
    ).model_dump(mode="json")

    def raise_error(*args):
        raise Exception("simulated parse failure")

    with monkeypatch.context() as mp:
        mp.setattr("preprocessors.router.create_vault", lambda _: directory_access)
        mp.setattr("preprocessors.router.parse_scan_pipeline", raise_error)
        response = send_post_request_with_mask(client=client, endpoint=endpoint, params=payload, mask=mask)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "could not parse scan file" in response.json()["detail"]


@pytest.mark.parametrize(
    ("endpoint", "schema", "mark_parameters", "mark_type", "controller_function"),
    _MARK_ENDPOINT_CASES,
)
def test_prepare_mark_returns_422_on_processing_error(
    client: TestClient,
    directory_access: DirectoryAccess,
    scan_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask: BinaryMask,
    endpoint: PreprocessorEndpoint,
    schema: type[PrepareMarkImpression | PrepareMarkStriation],
    mark_parameters: type[PreprocessingStriationParams | PreprocessingImpressionParams],
    mark_type: MarkType,
    controller_function: str,
) -> None:
    """Test that a 422 is returned when mark processing raises a ValueError."""
    payload = schema(
        project_name="test_project",
        mark_type=mark_type,
        scan_file=scan_directory / "circle.x3p",
        mark_parameters=mark_parameters(),  # type: ignore
        bounding_box_list=[],
    ).model_dump(mode="json")

    def raise_value_error(*args, **kwargs):
        raise ValueError("processing failed: empty mask")

    with monkeypatch.context() as mp:
        mp.setattr("preprocessors.router.create_vault", lambda _: directory_access)
        mp.setattr(controller_function, raise_value_error)
        response = send_post_request_with_mask(client=client, endpoint=endpoint, params=payload, mask=mask)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "processing failed" in response.json()["detail"]


def test_edit_scan_returns_422_on_scan_parse_error(
    client: TestClient,
    directory_access: DirectoryAccess,
    scan_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask: BinaryMask,
) -> None:
    """Test that a 422 is returned when the scan file for edit-scan cannot be parsed."""
    params = EditImage(
        project_name="test",
        scan_file=scan_directory / "circle.x3p",
        cutoff_length=2 * micro,
        terms=SurfaceOptions.PLANE,
    ).model_dump(mode="json")

    def raise_error(*args):
        raise Exception("simulated parse failure")

    with monkeypatch.context() as mp:
        mp.setattr("preprocessors.router.create_vault", lambda _: directory_access)
        mp.setattr("preprocessors.router.parse_scan_pipeline", raise_error)
        response = send_post_request_with_mask(client=client, endpoint="edit-scan", params=params, mask=mask)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "could not parse scan file" in response.json()["detail"]


def test_edit_scan_returns_422_on_edit_error(
    client: TestClient,
    directory_access: DirectoryAccess,
    scan_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask: BinaryMask,
) -> None:
    """Test that a 422 is returned when edit_scan_image raises a ValueError."""
    params = EditImage(
        project_name="test",
        scan_file=scan_directory / "circle.x3p",
        cutoff_length=2 * micro,
        terms=SurfaceOptions.PLANE,
    ).model_dump(mode="json")

    def raise_value_error(*args, **kwargs):
        raise ValueError("edit processing failed")

    with monkeypatch.context() as mp:
        mp.setattr("preprocessors.router.create_vault", lambda _: directory_access)
        mp.setattr("preprocessors.router.edit_scan_image", raise_value_error)
        response = send_post_request_with_mask(client=client, endpoint="edit-scan", params=params, mask=mask)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "edit processing failed" in response.json()["detail"]


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

    mask = np.zeros(shape=(259, 259), dtype=np.bool_)
    mask[1:259, 1:259] = True

    params = EditImage(
        project_name="test",
        scan_file=scan_directory / "circle.x3p",
        cutoff_length=2 * micro,
        resampling_factor=0.5,
        terms=SurfaceOptions.PLANE,
        regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        crop=True,
    ).model_dump(mode="json")

    # Act
    with monkeypatch.context() as mp:
        mp.setattr("preprocessors.router.create_vault", lambda _: directory_access)
        response = send_post_request_with_mask(client=client, endpoint="edit-scan", params=params, mask=mask)
    # Assert
    expected_response = GeneratedImages(
        preview_image=HttpUrl(f"{base_url}/preview.png"),
        surface_map_image=HttpUrl(f"{base_url}/surface_map.png"),
    )
    assert response.status_code == HTTPStatus.OK, "endpoint is alive"
    response_model = GeneratedImages.model_validate(response.json())
    assert response_model == expected_response
    assert (directory / "preview.png").exists()
    assert (directory / "surface_map.png").exists()
