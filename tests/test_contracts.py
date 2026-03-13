import json
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
from conversion.data_formats import Mark
from conversion.profile_correlator import Profile
from pydantic import BaseModel, Field, HttpUrl
from requests import Response

from models import DirectoryAccess
from preprocessors.pipelines import parse_scan_pipeline
from settings import get_settings
from tests.helper_function import _save_impression_mark, _save_striation_mark_and_profile, make_cell


class RoutePrefix(StrEnum):
    COMPARATOR = "comparator"
    EXTRACTOR = "extractor"
    PREPROCESSOR = "preprocessor"
    PROCESSOR = "processor"


class TemplateResponse(BaseModel):
    """Simple template response."""

    message: str


class EndpointContractInterface(BaseModel):
    expected_input: dict
    expected_urls: dict[str, str] = Field(default_factory=dict)
    expected_fields: dict[str, Any] = Field(default_factory=dict)


type Interface = tuple[BaseModel, type[BaseModel]]


def send_post_request_with_mask(endpoint: str, params: dict, mask_raw: bytes) -> Response:
    return requests.post(
        f"{get_settings().base_url}/{RoutePrefix.PREPROCESSOR}/{endpoint}",
        data={"params": json.dumps(params, default=str)},
        files={"mask_data": ("mask.bin", mask_raw, "application/octet-stream")},
        timeout=5,
    )


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
            expected_input={
                "project_name": "forensic_analysis_2026",
                "scan_file": str((scan_directory / "circle.x3p").absolute()),
                "scale_x": "1",
                "scale_y": "1",
                "step_size": "1",
            },
            expected_urls={
                "preview_image": ".png",
                "surface_map_image": ".png",
                "scan_image": ".x3p",
            },
        )

    @pytest.fixture(scope="class")
    def edit_scan(self, scan_directory: Path) -> tuple[EndpointContractInterface, bytes]:
        """Create test data for edit-scan endpoint.

        Returns the post request data, expected response type, and mask bytes.
        """
        cutoff_length = 250
        scan_file = scan_directory / "Klein_non_replica_mode_X3P_Scratch.x3p"
        parsed_scan = parse_scan_pipeline(scan_file, 1, 1)
        return EndpointContractInterface(
            expected_input={
                "scan_file": scan_file,
                "cutoff_length": cutoff_length,
                "mask_is_bitpacked": False,
                "terms": "plane",
            },
            expected_urls={
                "preview_image": ".png",
                "surface_map_image": ".png",
            },
        ), np.ones(parsed_scan.data.shape, dtype=np.bool_).tobytes(order="C")

    @pytest.fixture(scope="class")
    def prepare_mark_impression(self, scan_directory: Path, mask_raw: bytes) -> tuple[EndpointContractInterface, bytes]:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        scan_file = scan_directory / "circle.x3p"
        return EndpointContractInterface(
            expected_input={
                "scan_file": str(scan_file.absolute()),
                "mark_type": "breech face impression mark",
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
            expected_urls={
                "preview_image": ".png",
                "surface_map_image": ".png",
                "mark_data": ".npz",
                "mark_meta": ".json",
                "processed_data": ".npz",
                "processed_meta": ".json",
                "leveled_data": ".npz",
                "leveled_meta": ".json",
            },
        ), mask_raw

    @pytest.fixture(scope="class")
    def prepare_mark_striation(self, scan_directory: Path, mask_raw: bytes) -> tuple[EndpointContractInterface, bytes]:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        scan_file = scan_directory / "circle.x3p"
        return EndpointContractInterface(
            expected_input={
                "scan_file": str(scan_file.absolute()),
                "mark_type": "aperture shear striation mark",
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
            expected_urls={
                "preview_image": ".png",
                "surface_map_image": ".png",
                "mark_data": ".npz",
                "mark_meta": ".json",
                "processed_data": ".npz",
                "processed_meta": ".json",
                "profile_data": ".npz",
            },
        ), mask_raw

    @pytest.fixture
    def calculate_score_impression(
        self, directory_access: DirectoryAccess, dummy_mark: Mark
    ) -> EndpointContractInterface:
        """
        Create test data for calculate-score-impression endpoint.

        Returns the post request data and expected response type.
        """
        _save_impression_mark(directory_access.resource_path, dummy_mark)
        return EndpointContractInterface(
            expected_input={
                "mark_dir_ref": str(directory_access.resource_path),
                "mark_dir_comp": str(directory_access.resource_path),
                "metadata_reference": {
                    "case_id": "case_1",
                    "firearm_id": "firearm_1",
                    "specimen_id": "specimen_1",
                    "measurement_id": "measurement_1",
                    "mark_id": "mark_1",
                },
                "metadata_compared": {
                    "case_id": "case_1",
                    "firearm_id": "firearm_1",
                    "specimen_id": "specimen_1",
                    "measurement_id": "measurement_1",
                    "mark_id": "mark_2",
                },
                "comparison_params": {
                    "cell_size": [25e-6, 25e-6],
                    "minimum_fill_fraction": 0.1,
                },
            },
            expected_urls={
                "comparison_overview": ".png",
                "leveled_reference_heatmap": ".png",
                "leveled_compared_heatmap": ".png",
                "filtered_reference_heatmap": ".png",
                "filtered_compared_heatmap": ".png",
                "cell_reference_heatmap": ".png",
                "cell_compared_heatmap": ".png",
                "cell_overlay": ".png",
                "cell_cross_correlation": ".png",
            },
            expected_fields={
                "cells": list,
            },
        )

    @pytest.fixture
    def calculate_score_striation(
        self, directory_access: DirectoryAccess, dummy_profile: Profile, dummy_mark: Mark
    ) -> EndpointContractInterface:
        """
        Create test data for calculate-score-striation endpoint.

        Returns the post request data and expected response type.
        """
        _save_striation_mark_and_profile(directory_access.resource_path, profile=dummy_profile, mark=dummy_mark)
        return EndpointContractInterface(
            expected_input={
                "mark_dir_ref": str(directory_access.resource_path),
                "mark_dir_comp": str(directory_access.resource_path),
                "metadata_reference": {
                    "case_id": "case_1",
                    "firearm_id": "firearm_1",
                    "specimen_id": "specimen_1",
                    "measurement_id": "measurement_1",
                    "mark_id": "mark_1",
                },
                "metadata_compared": {
                    "case_id": "case_1",
                    "firearm_id": "firearm_1",
                    "specimen_id": "specimen_1",
                    "measurement_id": "measurement_1",
                    "mark_id": "mark_2",
                },
            },
            expected_urls={
                "mark_ref_surfacemap": ".png",
                "mark_comp_surfacemap": ".png",
                "filtered_reference_heatmap": ".png",
                "comparison_overview": ".png",
                "mark_reference_aligned_data": ".npz",
                "mark_reference_aligned_meta": ".json",
                "mark_compared_aligned_data": ".npz",
                "mark_compared_aligned_meta": ".json",
                "mark_ref_preview": ".png",
                "mark_comp_preview": ".png",
                "similarity_plot": ".png",
                "filtered_compared_heatmap": ".png",
                "side_by_side_heatmap": ".png",
            },
            expected_fields={"comparison_results": dict},
        )

    @pytest.fixture
    def calculate_lr_impression(self, directory_access: DirectoryAccess, tmp_path: Path) -> EndpointContractInterface:
        """
        Create test data for calculate-score-impression endpoint.

        Returns the post request data and expected response type.
        """
        (lr_system := tmp_path / "lr_system").touch()
        return EndpointContractInterface(
            expected_input={
                "mark_dir_ref": str(directory_access.resource_path),
                "mark_dir_comp": str(directory_access.resource_path),
                "metadata_reference": {
                    "case_id": "case_1",
                    "firearm_id": "firearm_1",
                    "specimen_id": "specimen_1",
                    "measurement_id": "measurement_1",
                    "mark_id": "mark_1",
                },
                "metadata_compared": {
                    "case_id": "case_1",
                    "firearm_id": "firearm_1",
                    "specimen_id": "specimen_1",
                    "measurement_id": "measurement_1",
                    "mark_id": "mark_2",
                },
                "lr_system_path": str(lr_system),
                "user_id": "ABCDE",
                "date_report": "2000-01-01",
                "score": 1,
                "n_cells": 5,
                "cells": [
                    make_cell(
                        center_reference=(i * 1e-3, 0.0),
                        best_score=0.3,
                        cell_size=(1e-3, 1e-3),
                    ).model_dump()
                    for i in range(5)
                ],
            },
            expected_urls={"lr_overview_plot": ".png"},
            expected_fields={"lr": 0},
        )

    @pytest.fixture
    def calculate_lr_striation(
        self,
        directory_access: DirectoryAccess,
        tmp_path: Path,
    ) -> EndpointContractInterface:
        """
        Create test data for calculate-score-striation endpoint.

        Returns the post request data and expected response type.
        """
        (lr_system := tmp_path / "lr_system").touch()
        return EndpointContractInterface(
            expected_input={
                "mark_dir_ref": str(directory_access.resource_path),
                "mark_dir_comp": str(directory_access.resource_path),
                "metadata_reference": {
                    "case_id": "case_1",
                    "firearm_id": "firearm_1",
                    "specimen_id": "specimen_1",
                    "measurement_id": "measurement_1",
                    "mark_id": "mark_1",
                },
                "metadata_compared": {
                    "case_id": "case_1",
                    "firearm_id": "firearm_1",
                    "specimen_id": "specimen_1",
                    "measurement_id": "measurement_1",
                    "mark_id": "mark_2",
                },
                "lr_system_path": str(lr_system),
                "user_id": "ABCDE",
                "date_report": "2000-01-01",
                "mark_dir_ref_aligned": str(directory_access.resource_path),
                "mark_dir_comp_aligned": str(directory_access.resource_path),
                "score": 0.5,
            },
            expected_urls={"lr_overview_plot": ".png"},
            expected_fields={"lr": 0},
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

    def _assert_response(self, response: Response, data: EndpointContractInterface):
        files = {
            "png": b"\x89PNG\r\n\x1a\n",
            "x3p": b"PK\x03\x04",
            "npz": b"PK\x03\x04",
            "json": b"{",
        }
        body = response.json()
        assert response.status_code == HTTPStatus.OK, response.text

        expected_keys = data.expected_urls.keys() | data.expected_fields.keys()
        assert expected_keys == body.keys(), "response keys should match expected keys"

        for key, expected_type in data.expected_fields.items():
            assert isinstance(body[key], expected_type), f"{key}: expected {expected_type}, got {type(body[key])}"

        for key, expected_ext in data.expected_urls.items():
            url = body[key]
            assert HttpUrl(url), f"{key} should be a valid URL"
            file_response = requests.get(url, timeout=10)
            assert file_response.status_code == HTTPStatus.OK, f"{key} URL should be reachable"
            file_type = file_response.url.split("/")[-1].split(".")[-1]
            assert expected_ext == f".{file_type}", f"{key}: expected {expected_ext}, got .{file_type}"
            assert file_response.content[: len(files[file_type])] == files[file_type], (
                f"{key}: file content should start with {files[file_type]}"
            )

    def test_pre_processor_post_requests_process_scan(self, request: pytest.FixtureRequest) -> None:
        """Test if preprocessor POST endpoint return expected models."""
        data: EndpointContractInterface = request.getfixturevalue("process_scan")
        # Act
        response = requests.post(
            f"{get_settings().base_url}/{RoutePrefix.PREPROCESSOR}/process-scan",
            json=data.expected_input,
            timeout=20,
        )
        # Assert
        self._assert_response(data=data, response=response)

    @pytest.mark.parametrize(
        ("fixture_name", "sub_route"),
        [
            pytest.param("prepare_mark_impression", "prepare-mark-impression", id="prepare_mark_impression"),
            pytest.param("prepare_mark_striation", "prepare-mark-striation", id="prepare_mark_striation"),
            pytest.param("edit_scan", "edit-scan", id="edit_scan"),
        ],
    )
    def test_pre_processor_post_requests_with_masks(
        self, fixture_name: str, sub_route: str, request: pytest.FixtureRequest
    ) -> None:
        """Test if preprocessor POST endpoints return expected models."""
        data: tuple[EndpointContractInterface, bytes] = request.getfixturevalue(fixture_name)
        params, mask_raw = data
        # Act
        response = send_post_request_with_mask(endpoint=sub_route, params=params.expected_input, mask_raw=mask_raw)
        # Assert
        self._assert_response(data=params, response=response)

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
                marks=pytest.mark.xfail(reason="requires valid LR system"),
            ),
            pytest.param(
                "calculate_lr_striation",
                "calculate-lr-striation",
                id="calculate_lr_striation",
                marks=pytest.mark.xfail(reason="requires valid LR system"),
            ),
        ],
    )
    def test_processor_post_requests(self, fixture_name: str, sub_route: str, request: pytest.FixtureRequest) -> None:
        """Test if processor POST endpoints return expected models."""
        data: EndpointContractInterface = request.getfixturevalue(fixture_name)
        # Act
        response = requests.post(
            f"{get_settings().base_url}/{RoutePrefix.PROCESSOR}/{sub_route}",
            json=data.expected_input,
            timeout=5,
        )
        # Assert
        assert response.status_code == HTTPStatus.OK
        self._assert_response(data=data, response=response)

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
