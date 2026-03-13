from http import HTTPStatus
from pathlib import Path

import pytest
from conversion.data_formats import MarkMetadata
from fastapi.testclient import TestClient
from pydantic import HttpUrl

from constants import ProcessorEndpoint
from processors.schemas import (
    CalculateLRImpression,
    CalculateLRStriation,
    CalculateScore,
)
from tests.processors.conftest import assert_lr_response_valid


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
            "mark_ref_surfacemap",
            "mark_ref_preview",
            "mark_comp_surfacemap",
            "mark_comp_preview",
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


class TestCalculateLRImpression:
    @pytest.mark.integration
    def test_returns_lr_and_plot_url(
        self,
        client: TestClient,
        tmp_dir_api: None,
        impression_kwargs: dict,
    ) -> None:
        """Endpoint returns a float LR and a reachable plot URL."""
        json_data = CalculateLRImpression(**impression_kwargs).model_dump(mode="json")
        response = client.post(f"/processor/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}", json=json_data)
        assert_lr_response_valid(client, response)


class TestCalculateLRStriation:
    @pytest.mark.integration
    def test_returns_lr_and_plot_url(
        self,
        client: TestClient,
        tmp_dir_api: None,
        striation_kwargs: dict,
    ) -> None:
        """Endpoint returns a float LR and a reachable plot URL."""
        json_data = CalculateLRStriation(**striation_kwargs).model_dump(mode="json")
        response = client.post(f"/processor/{ProcessorEndpoint.CALCULATE_LR_STRIATION}", json=json_data)
        assert_lr_response_valid(client, response)
