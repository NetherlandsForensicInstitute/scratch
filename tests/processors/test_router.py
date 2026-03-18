from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

import pytest
from conversion.data_formats import MarkMetadata
from conversion.likelihood_ratio import DummyLRSystem, ModelSpecs
from conversion.surface_comparison.models import ComparisonParams
from fastapi.testclient import TestClient
from pydantic import HttpUrl

from constants import ProcessorEndpoint
from processors.schemas import (
    CalculateLRImpression,
    CalculateLRStriation,
    CalculateScore,
    CalculateScoreImpression,
)

from ..helper_function import assert_lr_response_valid


def test_processors_placeholder(client: TestClient) -> None:
    """Test that the processor root endpoint redirects to documentation."""
    # Act
    response = client.get("/processor", follow_redirects=False)
    # Assert
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, "endpoint should redirect"
    assert response.headers["location"] == "/docs#operations-tag-processor", "should redirect to processor docs"


class TestMarkStriation:
    @pytest.mark.integration
    def test_calculate_striation_mark(
        self,
        client: TestClient,
        mark_dirs: tuple[Path, Path],
    ) -> None:
        """Test the whole chain of the endpoint."""
        mark_dir_ref, mark_dir_comp = mark_dirs
        expected_images = {
            "similarity_plot",
            "side_by_side_heatmap",
            "comparison_overview",
            "filtered_compared_heatmap",
            "filtered_reference_heatmap",
            "mark_reference_aligned_surfacemap",
            "mark_reference_aligned_preview",
            "mark_compared_aligned_surfacemap",
            "mark_compared_aligned_preview",
        }
        expected_data = {
            "mark_reference_aligned_meta",
            "mark_reference_aligned_data",
            "mark_compared_aligned_data",
            "mark_compared_aligned_meta",
            "comparison_results",
        }

        json_data = CalculateScore(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            metadata_compared=MarkMetadata(
                case_id="something",
                firearm_id="else",
                specimen_id="spec_comp",
                measurement_id="meas_comp",
                mark_id="mark_comp",
            ),
            metadata_reference=MarkMetadata(
                case_id="ding",
                firearm_id="dong",
                specimen_id="spec_ref",
                measurement_id="meas_ref",
                mark_id="mark_ref",
            ),
        ).model_dump(mode="json")
        response = client.post("/processor/" + ProcessorEndpoint.CALCULATE_SCORE_STRIATION, json=json_data)

        assert response.status_code == HTTPStatus.OK, response.json()
        response_data = response.json()
        assert response_data.keys() == (expected_images | expected_data)

        url_keys = (expected_images | expected_data) - {"comparison_results"}
        urls = {k: v for k, v in response_data.items() if k in url_keys}
        assert all(HttpUrl(url) for url in urls.values())
        assert all(client.get(url).status_code == HTTPStatus.OK for url in urls.values())

        for key in expected_images:
            assert client.get(urls[key]).headers["content-type"] == "image/png", f"{key} should be PNG"


class TestMarkImpression:
    @pytest.mark.integration
    def test_calculate_impression_mark(
        self,
        client: TestClient,
        impression_mark_dirs: tuple[Path, Path],
    ) -> None:
        """Test the whole chain of the impression comparison endpoint."""
        mark_dir_ref, mark_dir_comp = impression_mark_dirs
        expected_images = {
            "comparison_overview",
            "leveled_reference_heatmap",
            "leveled_compared_heatmap",
            "filtered_reference_heatmap",
            "filtered_compared_heatmap",
            "cell_reference_heatmap",
            "cell_compared_heatmap",
            "cell_overlay",
            "cell_cross_correlation",
        }
        expected_data = {
            "cells",
        }

        json_data = CalculateScoreImpression(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            comparison_params=ComparisonParams(
                cell_size=(50e-6, 50e-6),
                search_angle_min=-5.0,
                search_angle_max=5.0,
                search_angle_step=5.0,
                correlation_threshold=0.2,
                minimum_fill_fraction=0.1,
                angle_deviation_threshold=10.0,
                position_threshold=0.001,
            ),
            metadata_reference=MarkMetadata(
                case_id="case_ref",
                firearm_id="firearm_ref",
                specimen_id="spec_ref",
                measurement_id="meas_ref",
                mark_id="mark_ref",
            ),
            metadata_compared=MarkMetadata(
                case_id="case_comp",
                firearm_id="firearm_comp",
                specimen_id="spec_comp",
                measurement_id="meas_comp",
                mark_id="mark_comp",
            ),
        ).model_dump(mode="json")

        response = client.post(
            "/processor/" + ProcessorEndpoint.CALCULATE_SCORE_IMPRESSION,
            json=json_data,
        )

        assert response.status_code == HTTPStatus.OK, response.json()
        response_data = response.json()
        assert response_data.keys() == (expected_images | expected_data)

        # Verify cells exist
        assert len(response_data["cells"]) > 0

        # Verify image URLs
        url_keys = expected_images
        urls = {k: v for k, v in response_data.items() if k in url_keys}
        assert all(HttpUrl(url) for url in urls.values())
        assert all(client.get(url).status_code == HTTPStatus.OK for url in urls.values())

        for key in expected_images:
            assert client.get(urls[key]).headers["content-type"] == "image/png", f"{key} should be PNG"


class TestCalculateLRImpression:
    @pytest.mark.integration
    def test_returns_lr_and_plot_url(
        self, client: TestClient, tmp_dir_api: None, impression_kwargs: dict, impression_reference_data: ModelSpecs
    ) -> None:
        """Endpoint returns a float LR and a reachable plot URL."""
        with (
            patch("processors.controller.get_lr_system_from_path", return_value=DummyLRSystem()),
            patch("processors.controller.get_reference_data_from_path", return_value=impression_reference_data),
        ):
            json_data = CalculateLRImpression(**impression_kwargs).model_dump(mode="json")
            response = client.post(f"/processor/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}", json=json_data)
            assert_lr_response_valid(client, response)


class TestCalculateLRStriation:
    @pytest.mark.integration
    def test_returns_lr_and_plot_url(
        self, client: TestClient, tmp_dir_api: None, striation_kwargs: dict, striation_reference_data: ModelSpecs
    ) -> None:
        """Endpoint returns a float LR and a reachable plot URL."""
        with (
            patch("processors.controller.get_lr_system_from_path", return_value=DummyLRSystem()),
            patch("processors.controller.get_reference_data_from_path", return_value=striation_reference_data),
        ):
            json_data = CalculateLRStriation(**striation_kwargs).model_dump(mode="json")
            response = client.post(f"/processor/{ProcessorEndpoint.CALCULATE_LR_STRIATION}", json=json_data)
            assert_lr_response_valid(client, response)
