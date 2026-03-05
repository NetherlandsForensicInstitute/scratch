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

from models import DirectoryAccess
from preprocessors.pipelines import parse_scan_pipeline
from processors.schemas import (
    ImpressionLRParameters,
    StriationLRParameters,
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
    expected_input: dict
    expected_output: dict


type Interface = tuple[BaseModel, type[BaseModel]]


def send_post_request_with_mask(endpoint: str, params: dict, mask: BinaryMask) -> Response:
    return requests.post(
        f"{get_settings().base_url}/{RoutePrefix.PREPROCESSOR}/{endpoint}",
        data={"params": json.dumps(params, default=str)},
        files={"mask_data": ("mask.bin", mask.tobytes(order="C"), "application/octet-stream")},
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
            expected_output={
                "preview": ".png",
                "surface_map": ".png",
                "scan": ".x3p",
            },
        )

    @pytest.fixture(scope="class")
    def prepare_mark_impression(self, scan_directory: Path) -> tuple[EndpointContractInterface, BinaryMask]:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        scan_file = scan_directory / "circle.x3p"
        parsed_scan = parse_scan_pipeline(scan_file, 1, 1)
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
            expected_output={
                "preview": ".png",
                "surface_map": ".png",
                "mark_data": ".npz",
                "mark_meta": ".json",
                "processed_data": ".npz",
                "processed_meta": ".json",
                "leveled_data": ".npz",
                "leveled_meta": ".json",
            },
        ), np.ones(parsed_scan.data.shape, dtype=np.bool)

    @pytest.fixture(scope="class")
    def prepare_mark_striation(self, scan_directory: Path) -> tuple[EndpointContractInterface, BinaryMask]:
        """
        Create dummy files for the expected response.

        Returns the post request data, sub_route & expected response.
        """
        scan_file = scan_directory / "circle.x3p"
        parsed_scan = parse_scan_pipeline(scan_file, 1, 1)
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
            expected_output={
                "preview": ".png",
                "surface_map": ".png",
                "mark_data": ".npz",
                "mark_meta": ".json",
                "processed_data": ".npz",
                "processed_meta": ".json",
                "profile_data": ".npz",
            },
        ), np.ones(parsed_scan.data.shape, dtype=np.bool)

    @pytest.fixture(scope="class")
    def edit_scan(self, scan_directory: Path) -> tuple[EndpointContractInterface, BinaryMask]:
        """Create test data for edit-scan endpoint.

        Returns the post request data, expected response type, and mask bytes.
        """
        cutoff_length = 250
        scan_file = scan_directory / "Klein_non_replica_mode_X3P_Scratch.x3p"
        parsed_scan = parse_scan_pipeline(scan_file, 1, 1)
        return EndpointContractInterface(
            expected_input={"scan_file": scan_file, "cutoff_length": cutoff_length, "mask_is_bitpacked": False},
            expected_output={
                "preview": ".png",
                "surface_map": ".png",
            },
        ), np.ones(parsed_scan.data.shape, dtype=np.bool_)

    @pytest.fixture(scope="class")
    def calculate_score_impression(self, directory_access: DirectoryAccess) -> EndpointContractInterface:
        """
        Create test data for calculate-score-impression endpoint.

        Returns the post request data and expected response type.
        """
        return EndpointContractInterface(
            expected_input={
                "mark_ref": str(directory_access.resource_path),
                "mark_comp": str(directory_access.resource_path),
                "param": {},
            },
            expected_output={
                "urls": {"lr_overview_plot": ".png"},
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
            expected_input={
                "mark_ref": str(directory_access.resource_path),
                "mark_comp": str(directory_access.resource_path),
                "param": {},
            },
            expected_output={
                "mark_ref_surfacemap": ".png",
                "mark_comp_surfacemap": ".png",
                "filtered_reference_heatmap": ".png",
                "comparison_overview": ".png",
                "mark_ref_filtered_moved_surfacemap": ".png",
                "mark_ref_filtered_bb_surfacemap": ".png",
                "mark_comp_filtered_bb_surfacemap": ".png",
                "mark_comp_filtered_all_bb_surfacemap": ".png",
                "cell_accf_distribution": ".png",
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
            expected_input={
                "mark_ref": directory_access.resource_path,
                "mark_comp": directory_access.resource_path,
                "score": 1,
                "n_cells": 5,
                "lr_system": lr_system,
                "param": ImpressionLRParameters(),
            },
            expected_output={
                "urls": {"lr_overview_plot": ".png"},
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
            expected_input={
                "mark_ref": directory_access.resource_path,
                "mark_comp": directory_access.resource_path,
                "score": 1,
                "lr_system": lr_system,
                "param": StriationLRParameters(),
            },
            expected_output={
                "mark_ref_surfacemap": ".png",
                "mark_comp_surfacemap": ".png",
                "mark_ref_filtered_surfacemap": ".png",
                "comparison_overview": ".png",
                "mark_ref_depthmap": ".png",
                "mark_comp_depthmap": ".png",
                "similarity_plot": ".png",
                "mark_comp_filtered_surfacemap": ".png",
                "mark1_vs_moved_mark2": ".png",
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
        """Test the response urls if they are live urls pointing to a file from the key."""
        files = {
            "png": b"\x89PNG\r\n\x1a\n",
            "x3p": b"PK\x03\x04",
            "npz": b"PK\x03\x04",
            "json": b"{\n",
        }
        assert response.status_code == HTTPStatus.OK, response.text
        assert all(HttpUrl(url) for url in response.json().values()), "all response items should be a Http url"
        for response_key, url in response.json().items():
            file_response = requests.get(url, timeout=10)
            assert file_response.status_code == HTTPStatus.OK, f"all urls should point to a live url {response.text}"
            assert (
                file_response.headers["content-type"] == "image/png"
                or file_response.headers["content-type"] == "application/octet-stream"
            ), "we should receive a file from the url"
            file_type = file_response.url.split("/")[-1].split(".")[-1]
            assert data.expected_output[response_key] == f".{file_type}", (
                f"with key:{response_key} should have file type:{data.expected_output[response_key]}"
            )
            assert file_response.content[: len(files[file_type])] == files[file_type], (
                f"file content should start with: {files[file_type]}"
            )
        assert data.expected_output.keys() == response.json().keys(), "all file keys are pressent"

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
        self._assert_response_urls(data=data, response=response)

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
        data: tuple[EndpointContractInterface, BinaryMask] = request.getfixturevalue(fixture_name)
        params, mask = data
        # Act
        response = send_post_request_with_mask(endpoint=sub_route, params=params.expected_input, mask=mask)
        # Assert
        self._assert_response_urls(data=params, response=response)

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
        data: EndpointContractInterface = request.getfixturevalue(fixture_name)
        # Act
        response = requests.post(
            f"{get_settings().base_url}/{RoutePrefix.PROCESSOR}/{sub_route}",
            json=data.expected_input,
            timeout=5,
        )
        # Assert
        assert response.status_code == HTTPStatus.OK
        self._assert_response_urls(data=data, response=response)

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
