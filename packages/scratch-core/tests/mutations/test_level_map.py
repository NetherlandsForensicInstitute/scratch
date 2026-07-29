from pathlib import Path

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.data_formats import SurfaceTerms
from mutations.filter import LevelMap


@pytest.mark.integration
class TestLevelMapIntegration:
    RESOURCES_DIR = Path(__file__).parent.parent / "computations" / "resources"

    @pytest.mark.parametrize(
        "surface_terms, verified_file_name",
        [
            [SurfaceTerms.PLANE, "baseline_level_map_plane.npy"],
            [SurfaceTerms.SPHERE, "baseline_level_map_sphere.npy"],
        ],
    )
    def test_map_level(
        self,
        scan_image_with_nans: ScanImage,
        verified_file_name: str,
        surface_terms: SurfaceTerms,
    ):
        # Arrange
        verified = np.load(self.RESOURCES_DIR / verified_file_name)
        level_map_mutator = LevelMap(surface_terms=surface_terms)
        # Act
        result = level_map_mutator(scan_image_with_nans)
        # Assert
        assert np.allclose(result.data, verified, equal_nan=True)

    def test_map_level_none(self, scan_image_with_nans: ScanImage):
        # Arrange
        level_map_mutator = LevelMap(surface_terms=SurfaceTerms.NONE)
        # Act
        result = level_map_mutator(scan_image_with_nans)
        # Assert
        assert np.allclose(result.data, scan_image_with_nans.data, equal_nan=True)
