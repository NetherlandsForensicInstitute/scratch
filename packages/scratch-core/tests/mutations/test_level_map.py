from pathlib import Path
from container_models.base import Coordinate, Pair
from container_models.image import ImageContainer
from mutations.filter import LevelMap
import pytest
import numpy as np
from conversion.leveling.data_types import SurfaceTerms


@pytest.fixture
def mask_image_with_nans(image_with_nans: ImageContainer) -> ImageContainer:
    return ImageContainer(data=image_with_nans.data, metadata=image_with_nans.metadata)


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
        mask_image_with_nans: ImageContainer,
        verified_file_name: str,
        terms: SurfaceTerms,
    ):
        # Arrange
        verified = np.load(self.RESOURCES_DIR / verified_file_name)
        level_map_mutator = LevelMap(reference=mask_image_with_nans.center, terms=terms)
        # Act
        result = level_map_mutator(mask_image_with_nans).unwrap()
        # Assert
        assert np.allclose(result.data, verified, equal_nan=True)

    def test_map_level_none(self, mask_image_with_nans: ImageContainer):
        # Arrange
        level_map_mutator = LevelMap(
            reference=mask_image_with_nans.center, terms=SurfaceTerms.NONE
        )
        # Act
        result = level_map_mutator(mask_image_with_nans).unwrap()
        # Assert
        assert np.allclose(result.data, mask_image_with_nans.data, equal_nan=True)

    def test_map_level_offset(self, mask_image_with_nans: ImageContainer):
        # Arrange
        level_map_mutator = LevelMap(
            reference=mask_image_with_nans.center, terms=SurfaceTerms.OFFSET
        )
        # Act
        result = level_map_mutator(mask_image_with_nans).unwrap()
        # Assert
        assert np.isclose(np.nanmean(result.data), 0.0)
        assert np.allclose(
            result.data + np.nanmean(mask_image_with_nans.data),
            mask_image_with_nans.data,
            equal_nan=True,
        )

    @pytest.mark.parametrize(
        "terms, ref_point",
        [
            [SurfaceTerms.NONE, Pair(10.5, -5.2)],
            [SurfaceTerms.PLANE, Pair(10.5, -5.2)],
            [SurfaceTerms.SPHERE, Pair(10.5, -5.2)],
            [SurfaceTerms.OFFSET, Pair(10.5, -5.2)],
            [SurfaceTerms.DEFOCUS, Pair(1234.567, 1234.567)],
            [SurfaceTerms.ASTIG_45, Pair(1234.567, 1234.567)],
        ],
    )
    def test_map_level_reference_point_has_no_effect(
        self,
        mask_image_with_nans: ImageContainer,
        terms: SurfaceTerms,
        ref_point: Coordinate,
    ):
        # Arrange
        level_map_mutator = LevelMap(reference=mask_image_with_nans.center, terms=terms)
        Level_map_ref = LevelMap(reference=ref_point, terms=terms)
        # Act
        result_centered = level_map_mutator(mask_image_with_nans).unwrap()
        result_ref = Level_map_ref(mask_image_with_nans).unwrap()
        # Assert
        assert np.allclose(result_centered.data, result_ref.data, equal_nan=True)
