from pathlib import Path

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms
from mutations.filter import LevelMap


@pytest.mark.integration
class TestLevelMapIntegration:
    RESOURCES_DIR = (
        Path(__file__).parent.parent / "conversion" / "leveling" / "resources"
    )

    @pytest.mark.parametrize(
        "terms, verified_file_name",
        [
            [SurfaceTerms.PLANE, "baseline_level_map_plane.npy"],
            [SurfaceTerms.SPHERE, "baseline_level_map_sphere.npy"],
        ],
    )
    def test_map_level(
        self,
        scan_image_with_nans: ScanImage,
        verified_file_name: str,
        terms: SurfaceTerms,
    ):
        # Arrange
        verified = np.load(self.RESOURCES_DIR / verified_file_name)
        level_map_mutator = LevelMap(terms=terms)
        # Act
        result = level_map_mutator(scan_image_with_nans)
        # Assert
        assert np.allclose(result.data, verified, equal_nan=True)

    def test_map_level_none(self, scan_image_with_nans: ScanImage):
        # Arrange
        level_map_mutator = LevelMap(terms=SurfaceTerms.NONE)
        # Act
        result = level_map_mutator(scan_image_with_nans)
        # Assert
        assert np.allclose(result.data, scan_image_with_nans.data, equal_nan=True)

    def test_map_level_offset(self, scan_image_with_nans: ScanImage):
        # Arrange
        level_map_mutator = LevelMap(terms=SurfaceTerms.OFFSET)
        # Act
        result = level_map_mutator(scan_image_with_nans)
        # Assert
        assert np.isclose(np.nanmean(result.data), 0.0)
        assert np.allclose(
            result.data + np.nanmean(scan_image_with_nans.data),
            scan_image_with_nans.data,
            equal_nan=True,
        )
