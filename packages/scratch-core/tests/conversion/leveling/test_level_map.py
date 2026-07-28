from computations.constants import SurfaceTerms
from computations.spatial import level_map
from container_models.scan_image import ScanImage
import pytest
import numpy as np
from .constants import RESOURCES_DIR


@pytest.mark.integration
def test_map_level_sphere(scan_image_with_nans: ScanImage):
    verified = np.load(RESOURCES_DIR / "baseline_level_map_sphere.npy")
    result = level_map(scan_image_with_nans, SurfaceTerms.SPHERE)
    assert result
    assert np.allclose(result.leveled_map, verified, equal_nan=True)


@pytest.mark.integration
def test_map_level_plane(scan_image_with_nans: ScanImage):
    verified = np.load(RESOURCES_DIR / "baseline_level_map_plane.npy")
    result = level_map(scan_image_with_nans, SurfaceTerms.PLANE)
    assert result
    assert np.allclose(result.leveled_map, verified, equal_nan=True)


@pytest.mark.integration
def test_map_level_none_has_no_effect(scan_image_with_nans: ScanImage):
    result = level_map(scan_image_with_nans, SurfaceTerms.NONE)
    assert result
    assert np.allclose(result.leveled_map, scan_image_with_nans.data, equal_nan=True)
