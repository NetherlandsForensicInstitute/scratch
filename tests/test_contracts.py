import json
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pytest
import requests
from container_models.base import BinaryMask
from pydantic import BaseModel, HttpUrl
from requests import Response

from constants import PROJECT_ROOT
from extractors.schemas import (
    ComparisonResponseImpression,
    ComparisonResponseStriation,
    GeneratedImages,
    LRResponse,
    PrepareMarkResponseImpression,
    PrepareMarkResponseStriation,
    ProcessedDataAccess,
)
from models import DirectoryAccess
from preprocessors.pipelines import parse_scan_pipeline
from preprocessors.schemas import (
    EditImage,
    MaskParameters,
    PrepareMarkImpression,
    PrepareMarkStriation,
    PreprocessingImpressionParams,
    PreprocessingStriationParams,
    UploadScan,
)
from processors.schemas import (
    CalculateLRImpression,
    CalculateLRStriation,
    CalculateScoreImpression,
    CalculateScoreStriation,
    ImpressionLRParamaters,
    ImpressionParameters,
    StriationLRParamaters,
    StriationParamaters,
)
from settings import get_settings

class RoutePrefix(StrEnum):
    COMPARATOR = "comparator"
    EXTRACTOR = "extractor"
    PREPROCESSOR = "preprocessor"
    PROCESSOR = "processor"


class TemplateResponse(BaseModel):
    """Simple template response,."""

    message: str


class EndpointContractInterface(BaseModel):
    input_json: dict
    response_json: dict


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

    BASE_URL = "http://127.0.0.1:8000"

    @pytest.fixture(scope="class")
    def process_scan(self, scan_directory: Path) -> EndpointContractInterface:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        return EndpointContractInterface(
            input_json={
                "project_name": "forensic_analysis_2026",
                "scan_file": str((scan_directory / "circle.x3p").absolute()),
                "scale_x": "1",
                "scale_y": "1",
                "step_size": "1",
            },
            response_json={
                "preview": f"{self.BASE_URL}/extractor/files/GENERATED_KEY/preview.png",
                "surface_map": f"{self.BASE_URL}/extractor/files/GENERATED_KEY/surface_map.png",
                "scan": f"{self.BASE_URL}/extractor/files/GENERATED_KEY/scan.x3p",
            },
        )

    @pytest.fixture(scope="class")
    def prepare_mark_impression(self, scan_directory: Path, mask: BinaryMask) -> EndpointContractInterface:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        return EndpointContractInterface(
            input_json={
                "scan_file": str((scan_directory / "circle.x3p").absolute()),
                "mark_type": "breech face impression mark",
                "mask": mask,
                "bounding_box_list": [[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]],
                "mark_parameters": {
                    "pixel_size": None,
                    "adjust_pixel_spacing": True,
                    "level_offset": True,
                    "level_tilt": True,
                    "level_2nd": True,
                    "interp_method": "cubic",
                    "highpass_cutoff": 250.0e-6,
                    "lowpass_cutoff": 5.0e-6,
                    "highpass_regression_order": 2,
                    "lowpass_regression_order": 0,
                },
            },
            response_json={
                "preview": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/preview.png",
                "surface_map": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/surface_map.png",
                "mark_data": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark.npz",
                "mark_meta": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark.json",
                "processed_data": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/processed.npz",
                "processed_meta": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/processed.json",
                "leveled_data": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/leveled.npz",
                "leveled_meta": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/leveled.json",
            },
        )

    @pytest.fixture(scope="class")
    def prepare_mark_striation(self, scan_directory: Path, mask: BinaryMask) -> EndpointContractInterface:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        return EndpointContractInterface(
            input_json={
                "scan_file": str((scan_directory / "circle.x3p").absolute()),
                "mark_type": "aperture shear striation mark",
                "mask": mask,
                "bounding_box_list": [[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]],
                "mark_parameters": {
                    "highpass_cutoff": 2e-3,
                    "lowpass_cutoff": 2.5e-4,
                    "cut_borders_after_smoothing": True,
                    "use_mean": True,
                    "angle_accuracy": 0.1,
                    "max_iter": 25,
                    "subsampling_factor": 1,
                },
            },
            response_json={
                "preview": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/preview.png",
                "surface_map": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/surface_map.png",
                "mark_data": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark.npz",
                "mark_meta": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark.json",
                "processed_data": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/processed.npz",
                "processed_meta": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/processed.json",
                "profile_data": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/profile.npz",
            },
        )

    @pytest.fixture(scope="class")
    def edit_scan(self, scan_directory: Path) -> tuple[EndpointContractInterface,bytes]:
        """Create test data for edit-scan endpoint.

        Returns the post request data, expected response type, and mask bytes.
        """
        CUTOFF_LENGTH = 250
        scan_file = scan_directory / "Klein_non_replica_mode_X3P_Scratch.x3p"
        parsed_scan = parse_scan_pipeline(scan_file, 1, 1)
        rows, cols = parsed_scan.data.shape
        return EndpointContractInterface(
            input_json={
                "scan_file": scan_file,
                "cutoff_length": CUTOFF_LENGTH,
                "mask_parameters": {"shape": (rows, cols), "is_bitpacked": False},
            },
            response_json={
                "preview": "http://localhost:8000/preprocessor/files/surface_comparator_859lquto/preview.png",
                "surface_map": "http://localhost:8000/preprocessor/files/surface_comparator_859lquto/surface_map.png",
            },
        ),np.ones((rows, cols), dtype=np.bool_).tobytes(order="C")

    @pytest.fixture(scope="class")
    def calculate_score_impression(self, directory_access: DirectoryAccess) -> EndpointContractInterface:
        """
        Create test data for calculate-score-impression endpoint.

        Returns the post request data and expected response type.
        """
        return EndpointContractInterface(
            input_json={
                "mark_ref": str(directory_access.resource_path),
                "mark_comp": str(directory_access.resource_path),
                "param": {},
            },
            response_json={
                "urls": {"lr_overview_plot": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/profile.png"},
                "lr": 0,
            },
        )

    @pytest.fixture(scope="class")
    def calculate_score_striation(self, directory_access: DirectoryAccess) -> EndpointContractInterface:
        """
        Create test data for calculate-score-striation endpoint.

        Returns the post request data and expected response type.
        """
        return EndpointContractInterface(
            input_json={
                "mark_ref": str(directory_access.resource_path),
                "mark_comp": str(directory_access.resource_path),
                "param": {},
            },
            response_json={
                "mark_ref_surfacemap": f"{self.BASE_URL}/processor/files/GENERATED_KEY/mark_ref_surfacemap.png",
                "mark_comp_surfacemap": f"{self.BASE_URL}/processor/files/GENERATED_KEY/mark_comp_surfacemap.png",
                "filtered_reference_heatmap": f"{self.BASE_URL}/processor/files/GENERATED_KEY/filtered_reference_heatmap.png",
                "comparison_overview": f"{self.BASE_URL}/processor/files/GENERATED_KEY/comparison_overview.png",
                "mark_ref_filtered_moved_surfacemap": f"{self.BASE_URL}/processor/files/GENERATED_KEY/mark_ref_filtered_moved_surfacemap.png",
                "mark_ref_filtered_bb_surfacemap": f"{self.BASE_URL}/processor/files/GENERATED_KEY/mark_ref_filtered_bb_surfacemap.png",
                "mark_comp_filtered_bb_surfacemap": f"{self.BASE_URL}/processor/files/GENERATED_KEY/mark_comp_filtered_bb_surfacemap.png",
                "mark_comp_filtered_all_bb_surfacemap": f"{self.BASE_URL}/processor/files/GENERATED_KEY/mark_comp_filtered_all_bb_surfacemap.png",
                "cell_accf_distribution": f"{self.BASE_URL}/processor/files/GENERATED_KEY/cell_accf_distribution.png",
            },
        )

    @pytest.fixture(scope="class")
    def calculate_lr_impression(self, directory_access: DirectoryAccess, tmp_path: Path) -> EndpointContractInterface:
        """
        Create test data for calculate-score-impression endpoint.

        Returns the post request data and expected response type.
        """
        (lr_system := tmp_path / "lr_system").touch()
        return EndpointContractInterface(
            input_json={
                "mark_ref": directory_access.resource_path,
                "mark_comp": directory_access.resource_path,
                "score": 1,
                "n_cells": 5,
                "lr_system": lr_system,
                "param": ImpressionLRParamaters(),
            },
            response_json={
                "urls": {"lr_overview_plot": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/profile.png"},
                "lr": 0,
            },
        )

    @pytest.fixture(scope="class")
    def calculate_lr_striation(self, directory_access: DirectoryAccess, tmp_path: Path) -> EndpointContractInterface:
        """
        Create test data for calculate-score-striation endpoint.

        Returns the post request data and expected response type.
        """
        (lr_system := tmp_path / "lr_system").touch()
        return EndpointContractInterface(
            input_json={
                "mark_ref": directory_access.resource_path,
                "mark_comp": directory_access.resource_path,
                "score": 1,
                "lr_system": lr_system,
                "param": StriationLRParamaters(),
            },
            response_json={
                "mark_ref_surfacemap": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark_ref_surfacemap.png",
                "mark_comp_surfacemap": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark_comp_surfacemap.png",
                "mark_ref_filtered_surfacemap": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark_ref_filtered_surfacemap.png",
                "comparison_overview": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/comparison_overview.png",
                "mark_ref_depthmap": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark_ref_depthmap.png",
                "mark_comp_depthmap": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark_comp_depthmap.png",
                "similarity_plot": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/similarity_plot.png",
                "mark_comp_filtered_surfacemap": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark_comp_filtered_surfacemap.png",
                "mark1_vs_moved_mark2": f"{self.BASE_URL}/preprocessor/files/GENERATED_KEY/mark1_vs_moved_mark2.png",
            },
        )

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

    def _assert_response_urls(self, response: Response, data: EndpointContractInterface):
        assert response.status_code == HTTPStatus.OK, response.text
        assert all(HttpUrl(url) for url in response.json().values())
        for url in response.json().values():
            assert requests.get(url, timeout=10).status_code == HTTPStatus.OK, response.text
        assert data.response_json.keys() == response.json().keys()

    @pytest.mark.parametrize(
        ("fixture_name", "sub_route"),
        [
            pytest.param("process_scan", "process-scan", id="process_scan"),
            pytest.param("prepare_mark_impression", "prepare-mark-impression", id="prepare_mark_impression"),
            pytest.param("prepare_mark_striation", "prepare-mark-striation", id="prepare_mark_striation"),
        ],
    )
    def test_pre_processor_post_requests(
        self, fixture_name: str, sub_route: str, request: pytest.FixtureRequest
    ) -> None:
        """Test if preprocessor POST endpoints return expected models."""
        data: EndpointContractInterface = request.getfixturevalue(fixture_name)
        # Act
        response = requests.post(
            f"{get_settings().base_url}/{RoutePrefix.PREPROCESSOR}/{sub_route}",
            json=data.input_json,
            timeout=20,
        )
        # Assert
        self._assert_response_urls(data=data, response=response)

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
            pytest.param(
                "calculate_lr_impression",
                "calculate-lr-impression",
                id="calculate_lr_impression",
            ),
            pytest.param(
                "calculate_lr_striation",
                "calculate-lr-striation",
                id="calculate_lr_striation",
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

    def test_pre_processor_edit_image_post_requests(self, edit_scan: tuple[EndpointContractInterface,bytes]) -> None:
        """Test if preprocessor EditImage POST endpoints return expected models."""
        edit_scan_params, mask_bytes =edit_scan
        # Act
        response = requests.post(
            f"{get_settings().base_url}/{RoutePrefix.PREPROCESSOR}/edit-scan",
            data={"params": json.dumps(edit_scan_params.input_json, default=str)},
            files={"mask_data": ("mask.bin", mask_bytes, "application/octet-stream")},
            timeout=5,
        )
        # Assert
        self._assert_response_urls(data=edit_scan_params, response=response)

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
