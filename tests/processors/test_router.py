import json
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from container_models.base import DepthData
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.export.mark import ExportedMarkData
from conversion.profile_correlator import Profile
from fastapi.testclient import TestClient
from loguru import logger
from scipy.constants import micro

from constants import ProcessorEndpoint
from models import DirectoryAccess
from processors.schemas import CalculateScoreStriation, StriationParamaters


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
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        mark_type: MarkType = MarkType.EXTRACTOR_IMPRESSION,
        center: tuple[float, float] | None = None,
        meta_data: dict[str, Any] | None = None,
    ) -> Mark:
        """Create a Mark instance for testing."""
        scan_image = ScanImage(data=data, scale_x=scale_x, scale_y=scale_y)
        if meta_data is not None:
            return Mark(
                scan_image=scan_image,
                mark_type=mark_type,
                center=center,
                meta_data=meta_data,
            )
        return Mark(scan_image=scan_image, mark_type=mark_type, center=center)

    def _make_synthetic_striation_profile(
        self,
        n_samples: int = 1000,
        n_striations: int = 20,
        amplitude_um: float = 0.5,
        noise_level: float = 0.05,
        pixel_size_m: float = 0.5 * micro,
        seed: int | None = None,
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
        if seed is not None:
            np.random.seed(seed)

        # Create base profile with multiple frequency components
        x = np.linspace(0, n_striations * 2 * np.pi, n_samples)

        # Primary striation pattern
        data = np.sin(x) * amplitude_um * micro

        # Add some harmonics for realism
        data += np.sin(2 * x) * amplitude_um * 0.3 * micro
        data += np.sin(0.5 * x) * amplitude_um * 0.5 * micro

        # Add noise
        noise = np.random.normal(0, amplitude_um * noise_level * micro, n_samples)
        data += noise

        return Profile(heights=data, pixel_size=pixel_size_m)

    def _make_shifted_profile(
        self,
        profile: Profile,
        shift_samples: float,
        scale_factor: float = 1.0,
        seed: int | None = None,
    ) -> Profile:
        """
        Create a shifted and optionally scaled version of a profile.

        :param profile: Original profile.
        :param shift_samples: Number of samples to shift (can be fractional).
        :param scale_factor: Scaling factor (1.0 = no scaling).
        :param seed: Random seed for added noise.
        :returns: New Profile with shifted/scaled data.
        """
        from scipy.interpolate import interp1d

        if seed is not None:
            np.random.seed(seed)

        data = profile.heights
        n = len(data)

        # Create interpolator
        x_orig = np.arange(n)
        interpolator = interp1d(x_orig, data, kind="linear", fill_value=0, bounds_error=False)

        # Create new coordinates with shift and scale
        x_new = x_orig * scale_factor + shift_samples
        new_data = interpolator(x_new)

        # Add a small amount of noise
        new_data += np.random.normal(0, np.nanstd(data) * 0.01, n)

        return Profile(heights=new_data, pixel_size=profile.pixel_size)

    def _striation_mark(self, profile: Profile, n_cols: int = 50) -> Mark:
        """Build a striation Mark by tiling a profile across n_cols columns."""
        data = np.tile(profile.heights[:, np.newaxis], (1, n_cols))
        return self.make_mark(
            data,
            scale_x=profile.pixel_size,
            scale_y=profile.pixel_size,
            mark_type=MarkType.BULLET_GEA_STRIATION,
        )

    def test_calculate_striation_mark(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # Arrange
        comp_mark_path = tmp_path / "comp_mark"
        ref_mark_path = tmp_path / "ref_mark"
        comp_mark_path.mkdir(parents=True, exist_ok=True)
        ref_mark_path.mkdir(parents=True, exist_ok=True)

        profile_reference = self._make_synthetic_striation_profile(n_samples=1000, seed=42)
        profile_compared = self._make_shifted_profile(profile_reference, 10.0, seed=43)
        # write to npz
        profile_stem = "profile"
        np.savez(
            (ref_mark_path / profile_stem).with_suffix(".npz"),
            heights=profile_reference.heights,
            pixel_size=profile_reference.pixel_size,
        )

        np.savez(
            (comp_mark_path / profile_stem).with_suffix(".npz"),
            heights=profile_compared.heights,
            pixel_size=profile_compared.pixel_size,
        )
        # mark
        mark_stem = "processed"
        mark_reference = self._striation_mark(profile_reference)
        mark_compared = self._striation_mark(profile_compared)
        np.savez(
            (ref_mark_path / mark_stem).with_suffix(".npz"),
            data=mark_reference.scan_image.data,
        )

        np.savez(
            (comp_mark_path / mark_stem).with_suffix(".npz"),
            data=mark_compared.scan_image.data,
        )

        # Save metadata JSON
        ref_json = ExportedMarkData(
            mark_type=MarkType.EXTRACTOR_STRIATION,
            center=(mark_reference.scan_image.height, mark_reference.scan_image.width),
            scale_x=mark_reference.scan_image.scale_x,
            scale_y=mark_reference.scan_image.scale_y,
        ).model_dump(mode="json")
        ref_json["mark_type"] = "EXTRACTOR_STRIATION"  # quick fix, due to misfit in enum

        comp_json = ExportedMarkData(
            mark_type=MarkType.EXTRACTOR_STRIATION,
            center=(mark_compared.scan_image.height, mark_reference.scan_image.width),
            scale_x=mark_compared.scan_image.scale_x,
            scale_y=mark_compared.scan_image.scale_y,
        ).model_dump(mode="json")
        comp_json["mark_type"] = "EXTRACTOR_STRIATION"  # quick fix, due to misfit in enum
        with open((ref_mark_path / mark_stem).with_suffix(".json"), "w") as f:
            json.dump(ref_json, f)

        with open((comp_mark_path / mark_stem).with_suffix(".json"), "w") as f:
            json.dump(comp_json, f)

        json_data = CalculateScoreStriation(
            mark_ref=ref_mark_path,
            mark_comp=comp_mark_path,
            param=StriationParamaters(metadata_compared={"somthing": "else"}, metadata_reference={"ding": "dong"}),
        ).model_dump(mode="json")
        # Act
        response = client.post("/processor/" + ProcessorEndpoint.CALCULATE_SCORE_STRIATION, json=json_data)
        # Arrange
        assert response.status_code == 200, response.json()
