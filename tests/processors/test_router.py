import json
from datetime import date
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pytest
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.export.mark import ExportedMarkData
from conversion.profile_correlator import Profile
from fastapi.testclient import TestClient
from lir import InstanceData
from lir.data.models import FeatureData, LLRData
from lir.lrsystems.lrsystems import LRSystem
from pydantic import HttpUrl
from scipy.constants import micro
from scipy.interpolate import interp1d

from constants import ProcessorEndpoint
from processors.schemas import (
    CalculateLRImpression,
    CalculateLRStriation,
    CalculateScoreStriation,
    ImpressionLRParameters,
    StriationLRParameters,
    StriationParameters,
)


class _IdentityLRSystem(LRSystem):
    """Minimal LRSystem that returns the input score as the LLR."""

    def __init__(self) -> None:
        pass

    def apply(self, instances: InstanceData) -> LLRData:
        """Return the first feature column as LLR values."""
        assert isinstance(instances, FeatureData)
        return LLRData(features=instances.features[:, 0])


def test_processors_placeholder(client: TestClient) -> None:
    """Test that the processor root endpoint redirects to documentation."""
    # Act
    response = client.get("/processor", follow_redirects=False)
    # Assert
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, "endpoint should redirect"
    assert response.headers["location"] == "/docs#operations-tag-processor", "should redirect to processor docs"


def _create_dummy_profile(n_samples: int = 1000) -> Profile:
    """Create a synthetic striation profile for testing."""
    n_striations = 20
    amplitude_um = 0.5
    noise_level = 0.05
    pixel_size_m = 0.5 * micro

    x = np.linspace(0, n_striations * 2 * np.pi, n_samples)
    data = np.sin(x) * amplitude_um * micro
    data += np.sin(2 * x) * amplitude_um * 0.3 * micro
    data += np.sin(0.5 * x) * amplitude_um * 0.5 * micro

    rng = np.random.default_rng()
    data += rng.normal(0, amplitude_um * noise_level * micro, n_samples)

    return Profile(heights=data, pixel_size=pixel_size_m)


def _shift_profile(profile: Profile, shift_samples: float) -> Profile:
    """Create a shifted version of a profile."""
    data = profile.heights
    n = len(data)
    x_orig = np.arange(n)
    interpolator = interp1d(x_orig, data, kind="linear", fill_value=0, bounds_error=False)
    x_new = x_orig + shift_samples
    new_data = interpolator(x_new)

    rng = np.random.default_rng()
    new_data += rng.normal(0, np.nanstd(data) * 0.01, n)

    return Profile(heights=new_data, pixel_size=profile.pixel_size)


def _striation_mark(profile: Profile, n_cols: int = 50) -> Mark:
    """Build a striation Mark by tiling a profile across columns."""
    data = np.tile(profile.heights[:, np.newaxis], (1, n_cols))
    scan_image = ScanImage(data=data, scale_x=profile.pixel_size, scale_y=profile.pixel_size)
    return Mark(scan_image=scan_image, mark_type=MarkType.BULLET_GEA_STRIATION, center=None)


def _save_mark_and_profile(dir_path: Path, profile: Profile, mark: Mark) -> None:
    """Save mark and profile files to a directory."""
    np.savez(dir_path / "profile.npz", heights=profile.heights, pixel_size=profile.pixel_size)
    np.savez(dir_path / "processed.npz", data=mark.scan_image.data)
    mark_json = ExportedMarkData(
        mark_type=MarkType.EXTRACTOR_STRIATION,
        center=(mark.scan_image.height, mark.scan_image.width),
        scale_x=mark.scan_image.scale_x,
        scale_y=mark.scan_image.scale_y,
    ).model_dump(mode="json")
    mark_json["mark_type"] = "EXTRACTOR_STRIATION"
    with open(dir_path / "processed.json", "w") as f:
        json.dump(mark_json, f)


@pytest.fixture
def mark_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Prepare directories with striation mark and profile files."""
    ref_path = tmp_path / "ref_mark"
    comp_path = tmp_path / "comp_mark"
    ref_path.mkdir()
    comp_path.mkdir()

    profile_ref = _create_dummy_profile()
    profile_comp = _shift_profile(profile_ref, 10.0)
    mark_ref = _striation_mark(profile_ref)
    mark_comp = _striation_mark(profile_comp)

    _save_mark_and_profile(ref_path, profile_ref, mark_ref)
    _save_mark_and_profile(comp_path, profile_comp, mark_comp)

    return ref_path, comp_path


class TestMarkStriation:
    @pytest.mark.integration
    def test_calculate_striation_mark(
        self,
        client: TestClient,
        mark_dirs: tuple[Path, Path],
    ) -> None:
        """
        Test the whole chain of the endpoint.

        The test expects a folder with some json/npz files.
        Those files containing information like the preprocces scan_image.
        """
        # Arrange
        mark_dir_comp, mark_dir_ref = mark_dirs
        expected_files = [
            "similarity_plot",
            "side_by_side_heatmap",
            "comparison_overview",
            "filtered_compared_heatmap",
            "filtered_reference_heatmap",
            "mark_ref_surfacemap",
            "mark_ref_preview",
            "mark_comp_surfacemap",
            "mark_comp_preview",
        ]

        json_data = CalculateScoreStriation(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            param=StriationParameters(metadata_compared={"something": "else"}, metadata_reference={"ding": "dong"}),
        ).model_dump(mode="json")

        # Act
        response = client.post("/processor/" + ProcessorEndpoint.CALCULATE_SCORE_STRIATION, json=json_data)

        # Arrange
        assert response.status_code == HTTPStatus.OK, response.json()
        urls = response.json()
        assert all(HttpUrl(url) for url in urls.values()), "All items in the response should be an url."
        assert all(url in expected_files for url in urls.keys()), "All expected files are in the url response."
        assert all(client.get(url).status_code == HTTPStatus.OK for url in urls.values()), (
            "Urls point to an living endpoint."
        )
        assert all(client.get(url).headers["content-type"] == "image/png" for url in urls.values()), (
            "Urls are returning an image."
        )


class TestCalculateLRImpression:
    @pytest.mark.integration
    def test_returns_lr_and_plot_url(  # noqa: PLR0913
        self,
        client: TestClient,
        tmp_dir_api: None,
        lr_system_path: Path,
        mark_dir_ref: Path,
        mark_dir_comp: Path,
        mark_dirs,
    ) -> None:
        """Endpoint returns a float LR and a reachable plot URL."""
        mark_dir_comp, mark_dir_ref = mark_dirs
        json_data = CalculateLRImpression(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            score=3,
            n_cells=10,
            lr_system_path=lr_system_path,
            param=ImpressionLRParameters(),
            metadata_compared={"metadata": "compared"},
            metadata_reference={"metadata": "reference"},
            user_id="AAAAA",
            date_report=date(2000, 1, 1),
        ).model_dump(mode="json")

        response = client.post(f"/processor/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}", json=json_data)

        assert response.status_code == HTTPStatus.OK, response.json()
        data = response.json()
        assert isinstance(data["lr"], float)
        assert HttpUrl(data["lr_overview_plot"])
        assert client.get(data["lr_overview_plot"]).status_code == HTTPStatus.OK
        assert client.get(data["lr_overview_plot"]).headers["content-type"] == "image/png"


class TestCalculateLRStriation:
    @pytest.mark.integration
    def test_returns_lr_and_plot_url(
        self,
        client: TestClient,
        tmp_dir_api: None,
        lr_system_path: Path,
        mark_dirs: tuple[Path, Path],
    ) -> None:
        """Endpoint returns a float LR and a reachable plot URL."""
        mark_dir_comp, mark_dir_ref = mark_dirs
        json_data = CalculateLRStriation(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            score=0.5,
            lr_system_path=lr_system_path,
            param=StriationLRParameters(),
            metadata_compared={"metadata": "compared"},
            metadata_reference={"metadata": "reference"},
            user_id="AAAAA",
            date_report=date(2000, 1, 1),
        ).model_dump(mode="json")

        response = client.post(f"/processor/{ProcessorEndpoint.CALCULATE_LR_STRIATION}", json=json_data)

        assert response.status_code == HTTPStatus.OK, response.json()
        data = response.json()
        assert isinstance(data["lr"], float)
        assert HttpUrl(data["lr_overview_plot"])
        assert client.get(data["lr_overview_plot"]).status_code == HTTPStatus.OK
        assert client.get(data["lr_overview_plot"]).headers["content-type"] == "image/png"
