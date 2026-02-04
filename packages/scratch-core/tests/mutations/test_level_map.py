from pathlib import Path
from mutations.filter import LevelMap
import pytest
import numpy as np
from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms


class TestLevelMap:
    RESOURCES_DIR = (
        Path(__file__).parent.parent / "conversion" / "leveling" / "resources"
    )

    def compute_image_center(self, scan_image: ScanImage) -> tuple[float, float]:
        """Compute the centerpoint (Y, X) of a scan image in physical coordinate space."""
        center_x = (scan_image.width - 1) * scan_image.scale_x * 0.5
        center_y = (scan_image.height - 1) * scan_image.scale_y * 0.5
        return center_y, center_x

    @pytest.mark.integration
    def test_map_level_sphere(self, scan_image_with_nans: ScanImage):
        # Arrange
        verified = np.load(self.RESOURCES_DIR / "baseline_level_map_sphere.npy")
        ycor, xcor = self.compute_image_center(scan_image=scan_image_with_nans)
        Level_map_mutator = LevelMap(
            x_reference_point=xcor, y_reference_point=ycor, terms=SurfaceTerms.SPHERE
        )
        # Act
        result = Level_map_mutator(scan_image_with_nans).unwrap()
        # Assert
        assert np.allclose(result.data, verified, equal_nan=True)

    @pytest.mark.integration
    def test_map_level_plane(self, scan_image_with_nans: ScanImage):
        # Arrange
        verified = np.load(self.RESOURCES_DIR / "baseline_level_map_plane.npy")
        ycor, xcor = self.compute_image_center(scan_image=scan_image_with_nans)
        Level_map_mutator = LevelMap(
            x_reference_point=xcor, y_reference_point=ycor, terms=SurfaceTerms.PLANE
        )
        # Act
        result = Level_map_mutator(scan_image_with_nans).unwrap()
        # Assert
        assert np.allclose(result.data, verified, equal_nan=True)

    @pytest.mark.integration
    def test_map_level_none(self, scan_image_with_nans: ScanImage):
        # Arrange
        ycor, xcor = self.compute_image_center(scan_image=scan_image_with_nans)
        Level_map_mutator = LevelMap(
            x_reference_point=xcor, y_reference_point=ycor, terms=SurfaceTerms.NONE
        )
        result = Level_map_mutator(scan_image_with_nans).unwrap()
        assert np.allclose(result.data, scan_image_with_nans.data, equal_nan=True)

    @pytest.mark.integration
    def test_map_level_offset(self, scan_image_with_nans: ScanImage):
        # Arrange
        ycor, xcor = self.compute_image_center(scan_image=scan_image_with_nans)
        Level_map_mutator = LevelMap(
            x_reference_point=xcor, y_reference_point=ycor, terms=SurfaceTerms.OFFSET
        )
        # Act
        result = Level_map_mutator(scan_image_with_nans).unwrap()
        # Assert
        assert np.isclose(np.nanmean(result.data), 0.0)
        assert np.allclose(
            result.data + np.nanmean(scan_image_with_nans.data),
            scan_image_with_nans.data,
            equal_nan=True,
        )

    @pytest.mark.parametrize(
        "terms, ref_point",
        [
            [SurfaceTerms.NONE, (10.5, -5.2)],
            [SurfaceTerms.PLANE, (10.5, -5.2)],
            [SurfaceTerms.SPHERE, (10.5, -5.2)],
            [SurfaceTerms.OFFSET, (10.5, -5.2)],
            [SurfaceTerms.DEFOCUS, (1234.567, 1234.567)],
            [SurfaceTerms.PLANE, (1234.567, 1234.567)],
            [SurfaceTerms.SPHERE, (1234.567, 1234.567)],
            [SurfaceTerms.ASTIG_45, (1234.567, 1234.567)],
        ],
    )
    def test_map_level_reference_point_has_no_effect(
        self, scan_image_with_nans: ScanImage, terms: SurfaceTerms, ref_point
    ):
        # Arrange
        ycor, xcor = self.compute_image_center(scan_image=scan_image_with_nans)
        Level_map_mutator = LevelMap(
            x_reference_point=xcor, y_reference_point=ycor, terms=terms
        )
        Level_map_ref = LevelMap(
            x_reference_point=ref_point[0], y_reference_point=ref_point[1], terms=terms
        )

        result_centered = Level_map_mutator(scan_image_with_nans).unwrap()
        result_ref = Level_map_ref(scan_image_with_nans).unwrap()
        assert np.allclose(result_centered.data, result_ref.data, equal_nan=True)
