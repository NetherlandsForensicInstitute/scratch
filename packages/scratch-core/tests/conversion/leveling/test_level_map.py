from conversion.leveling import level_map, SurfaceTerms
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
def test_map_level_none(scan_image_with_nans: ScanImage):
    result = level_map(scan_image_with_nans, SurfaceTerms.NONE)
    assert result
    assert np.allclose(result.leveled_map, scan_image_with_nans.data, equal_nan=True)


@pytest.mark.integration
def test_map_level_offset(scan_image_with_nans: ScanImage):
    result = level_map(scan_image_with_nans, SurfaceTerms.OFFSET)
    assert result
    assert np.isclose(np.nanmean(result.leveled_map), 0.0)
    assert np.allclose(
        result.leveled_map + np.nanmean(scan_image_with_nans.data),
        scan_image_with_nans.data,
        equal_nan=True,
    )
