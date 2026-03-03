import json
import pickle
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pytest
from container_models.base import DepthData
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.export.mark import ExportedMarkData
from conversion.profile_correlator import Profile
from fastapi.testclient import TestClient
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
    ImpressionLRParamaters,
    StriationLRParamaters,
    StriationParamaters,
)


class _IdentityLRSystem(LRSystem):
    """Minimal LRSystem that returns the input score as the LLR."""

    def __init__(self) -> None:
        super().__init__(name="identity")

    def apply(self, instances: FeatureData) -> LLRData:
        """Return input features as LLR values."""
        return LLRData(features=instances.features)


def test_processors_placeholder(client: TestClient) -> None:
    """Test that the processor root endpoint redirects to documentation."""
    # Act
    response = client.get("/processor", follow_redirects=False)
    # Assert
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, "endpoint should redirect"
    assert response.headers["location"] == "/docs#operations-tag-processor", "should redirect to processor docs"


class TestStriationMark:
    def make_mark(
        self,
        data: DepthData,
        scale_x: float,
        scale_y: float,
        center: tuple[float, float] | None,
        mark_type: MarkType = MarkType.EXTRACTOR_IMPRESSION,
    ) -> Mark:
        """Create a Mark instance for testing."""
        scan_image = ScanImage(data=data, scale_x=scale_x, scale_y=scale_y)
        return Mark(scan_image=scan_image, mark_type=mark_type, center=center)

    def _create_dummy_profile(
        self,
        n_samples: int,
    ) -> Profile:
        """
        Create a synthetic striation profile for testing.

        This function generates a profile that mimics the appearance of striation
        marks with multiple ridges and valleys.

        :param n_samples: Number of samples in the profile.
        :param n_striations: Number of striation features.
        :param amplitude_um: Amplitude of striations in micrometers.
        :param noise_level: Relative noise level (0 to 1).
        :param pixel_size_m: Pixel size in meters.
        :param seed: Random seed for reproducibility.
        :returns: Profile with synthetic striation data.
        """
        n_striations = 20
        amplitude_um = 0.5
        noise_level = 0.05
        pixel_size_m = 0.5 * micro

        # Create base profile with multiple frequency components
        x = np.linspace(0, n_striations * 2 * np.pi, n_samples)

        # Primary striation pattern
        data = np.sin(x) * amplitude_um * micro

        # Add some harmonics for realism
        data += np.sin(2 * x) * amplitude_um * 0.3 * micro
        data += np.sin(0.5 * x) * amplitude_um * 0.5 * micro

        # Add noise
        rng = np.random.default_rng()
        noise = rng.normal(0, amplitude_um * noise_level * micro, n_samples)
        data += noise

        return Profile(heights=data, pixel_size=pixel_size_m)

    def _shift_profile(
        self,
        profile: Profile,
        shift_samples: float,
    ) -> Profile:
        """
        Create a shifted and optionally scaled version of a profile.

        :param profile: Original profile.
        :param shift_samples: Number of samples to shift (can be fractional).
        :param scale_factor: Scaling factor (1.0 = no scaling).
        :param seed: Random seed for added noise.
        :returns: New Profile with shifted/scaled data.
        """
        scale_factor = (1.0,)
        data = profile.heights
        n = len(data)

        # Create interpolator
        x_orig = np.arange(n)
        interpolator = interp1d(x_orig, data, kind="linear", fill_value=0, bounds_error=False)

        # Create new coordinates with shift and scale
        x_new = x_orig * scale_factor + shift_samples
        new_data = interpolator(x_new)

        # Add a small amount of noise
        rng = np.random.default_rng()
        new_data += rng.normal(0, np.nanstd(data) * 0.01, n)

        return Profile(heights=new_data, pixel_size=profile.pixel_size)

    def _striation_mark(self, profile: Profile) -> Mark:
        """Build a striation Mark by tiling a profile across n_cols columns."""
        n_cols: int = 50
        data = np.tile(profile.heights[:, np.newaxis], (1, n_cols))
        return self.make_mark(
            data,
            scale_x=profile.pixel_size,
            scale_y=profile.pixel_size,
            mark_type=MarkType.BULLET_GEA_STRIATION,
            center=None,
        )

    def _save_profile(self, dir_path: Path, profile: Profile):
        np.savez(
            (dir_path / "profile").with_suffix(".npz"),
            heights=profile.heights,
            pixel_size=profile.pixel_size,
        )

    def _save_mark(self, dir_path: Path, profile: Profile, mark: Mark):
        mark_stem = "processed"
        mark_json = ExportedMarkData(
            mark_type=MarkType.EXTRACTOR_STRIATION,
            center=(mark.scan_image.height, mark.scan_image.width),
            scale_x=mark.scan_image.scale_x,
            scale_y=mark.scan_image.scale_y,
        ).model_dump(mode="json")
        mark_json["mark_type"] = "EXTRACTOR_STRIATION"  # quick fix, due to misfit in enum

        np.savez(
            (dir_path / mark_stem).with_suffix(".npz"),
            data=mark.scan_image.data,
        )
        with open((dir_path / mark_stem).with_suffix(".json"), "w") as f:
            json.dump(mark_json, f)

    @pytest.fixture
    def prepare_folder_for_calculation(self, tmp_path: Path) -> tuple[Path, Path]:
        """Prepare folder with marking and profiles for testing."""
        comp_mark_path = tmp_path / "comp_mark"
        ref_mark_path = tmp_path / "ref_mark"
        comp_mark_path.mkdir(parents=True, exist_ok=True)
        ref_mark_path.mkdir(parents=True, exist_ok=True)

        profile_reference = self._create_dummy_profile(n_samples=1000)
        profile_compared = self._shift_profile(profile_reference, 10.0)
        mark_reference = self._striation_mark(profile_reference)
        mark_compared = self._striation_mark(profile_compared)

        self._save_profile(ref_mark_path, profile_reference)
        self._save_profile(comp_mark_path, profile_compared)
        self._save_mark(dir_path=ref_mark_path, profile=profile_reference, mark=mark_reference)
        self._save_mark(dir_path=comp_mark_path, profile=profile_compared, mark=mark_compared)

        return (comp_mark_path, ref_mark_path)

    @pytest.mark.integration
    def test_calculate_striation_mark(
        self, client: TestClient, prepare_folder_for_calculation: tuple[Path, Path]
    ) -> None:
        """
        Test the whole chain of the endpoint.

        The test expects a folder with some json/npz files.
        Those files containing information like the preprocces scan_image.
        """
        # Arrange
        comp_mark_path, ref_mark_path = prepare_folder_for_calculation
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
            mark_dir_ref=ref_mark_path,
            mark_dir_comp=comp_mark_path,
            param=StriationParamaters(metadata_compared={"somthing": "else"}, metadata_reference={"ding": "dong"}),
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
    @pytest.fixture
    def lr_system_path(self, tmp_path: Path) -> Path:
        """Pickle an identity LRSystem and return its path."""
        path = tmp_path / "lr_system.pkl"
        path.write_bytes(pickle.dumps(_IdentityLRSystem()))
        return path

    @pytest.fixture
    def mark_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create empty mark directories."""
        ref = tmp_path / "mark_ref"
        comp = tmp_path / "mark_comp"
        ref.mkdir()
        comp.mkdir()
        return ref, comp

    @pytest.mark.integration
    def test_returns_lr_and_plot_url(
        self,
        client: TestClient,
        tmp_dir_api: None,
        lr_system_path: Path,
        mark_dirs: tuple[Path, Path],
    ) -> None:
        """Endpoint returns a float LR and a reachable plot URL."""
        ref_path, comp_path = mark_dirs
        json_data = CalculateLRImpression(
            mark_dir_ref=ref_path,
            mark_dir_comp=comp_path,
            score=3,
            n_cells=10,
            lr_system_path=lr_system_path,
            param=ImpressionLRParamaters(),
        ).model_dump(mode="json")

        response = client.post(f"/processor/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}", json=json_data)

        assert response.status_code == HTTPStatus.OK, response.json()
        data = response.json()
        assert isinstance(data["lr"], float)
        assert HttpUrl(data["lr_overview_plot"])
        assert client.get(data["lr_overview_plot"]).status_code == HTTPStatus.OK
        assert client.get(data["lr_overview_plot"]).headers["content-type"] == "image/png"


class TestCalculateLRStriation:
    @pytest.fixture
    def lr_system_path(self, tmp_path: Path) -> Path:
        """Pickle an identity LRSystem and return its path."""
        path = tmp_path / "lr_system.pkl"
        path.write_bytes(pickle.dumps(_IdentityLRSystem()))
        return path

    @pytest.fixture
    def mark_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create empty mark directories."""
        ref = tmp_path / "mark_ref"
        comp = tmp_path / "mark_comp"
        ref.mkdir()
        comp.mkdir()
        return ref, comp

    @pytest.mark.integration
    def test_returns_lr_and_plot_url(
        self,
        client: TestClient,
        tmp_dir_api: None,
        lr_system_path: Path,
        mark_dirs: tuple[Path, Path],
    ) -> None:
        """Endpoint returns a float LR and a reachable plot URL."""
        ref_path, comp_path = mark_dirs
        json_data = CalculateLRStriation(
            mark_dir_ref=ref_path,
            mark_dir_comp=comp_path,
            score=0.5,
            lr_system_path=lr_system_path,
            param=StriationLRParamaters(),
        ).model_dump(mode="json")

        response = client.post(f"/processor/{ProcessorEndpoint.CALCULATE_LR_STRIATION}", json=json_data)

        assert response.status_code == HTTPStatus.OK, response.json()
        data = response.json()
        assert isinstance(data["lr"], float)
        assert HttpUrl(data["lr_overview_plot"])
        assert client.get(data["lr_overview_plot"]).status_code == HTTPStatus.OK
        assert client.get(data["lr_overview_plot"]).headers["content-type"] == "image/png"
