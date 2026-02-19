from pathlib import Path
from mutations.filter import LevelMap
import pytest
import numpy as np
from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms


@pytest.mark.integration
class TestLevelMapIntegration:
    RESOURCES_DIR = (
        Path(__file__).parent.parent / "conversion" / "leveling" / "resources"
    )

    def compute_image_center(self, scan_image: ScanImage) -> tuple[float, float]:
        """Compute the centerpoint (Y, X) of a scan image in physical coordinate space."""
        center_x = (scan_image.width - 1) * scan_image.scale_x * 0.5
        center_y = (scan_image.height - 1) * scan_image.scale_y * 0.5
        return center_y, center_x

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
        y_center, x_center = self.compute_image_center(scan_image=scan_image_with_nans)
        level_map_mutator = LevelMap(terms=terms)
        # Act
        result = level_map_mutator(scan_image_with_nans).unwrap()
        # Assert
        assert np.allclose(result.data, verified, equal_nan=True)

    def test_map_level_none(self, scan_image_with_nans: ScanImage):
        # Arrange
        y_center, x_center = self.compute_image_center(scan_image=scan_image_with_nans)
        level_map_mutator = LevelMap(terms=SurfaceTerms.NONE)
        result = level_map_mutator(scan_image_with_nans).unwrap()
        assert np.allclose(result.data, scan_image_with_nans.data, equal_nan=True)

    def test_map_level_offset(self, scan_image_with_nans: ScanImage):
        # Arrange
        y_center, x_center = self.compute_image_center(scan_image=scan_image_with_nans)
        level_map_mutator = LevelMap(terms=SurfaceTerms.OFFSET)
        # Act
        result = level_map_mutator(scan_image_with_nans).unwrap()
        # Assert
        assert np.isclose(np.nanmean(result.data), 0.0)
        assert np.allclose(
            result.data + np.nanmean(scan_image_with_nans.data),
            scan_image_with_nans.data,
            equal_nan=True,
        )
