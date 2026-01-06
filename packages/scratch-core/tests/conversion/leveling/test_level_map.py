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
    assert all(
        np.isclose(result.parameters[p], 0.0)
        for p in SurfaceTerms
        if p not in SurfaceTerms.SPHERE
    )
    assert all(not np.isclose(result.parameters[p], 0.0) for p in SurfaceTerms.SPHERE)


@pytest.mark.integration
def test_map_level_plane(scan_image_with_nans: ScanImage):
    verified = np.load(RESOURCES_DIR / "baseline_level_map_plane.npy")
    result = level_map(scan_image_with_nans, SurfaceTerms.PLANE)
    assert result
    assert np.allclose(result.leveled_map, verified, equal_nan=True)
    assert all(
        np.isclose(result.parameters[p], 0.0)
        for p in SurfaceTerms
        if p not in SurfaceTerms.PLANE
    )
    assert all(not np.isclose(result.parameters[p], 0.0) for p in SurfaceTerms.PLANE)


@pytest.mark.integration
def test_map_level_none(scan_image_with_nans: ScanImage):
    result = level_map(scan_image_with_nans, SurfaceTerms.NONE)
    assert result
    assert np.allclose(result.leveled_map, scan_image_with_nans.data, equal_nan=True)
    assert all(np.isclose(result.parameters[p], 0.0) for p in SurfaceTerms)


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
    assert result.parameters[SurfaceTerms.OFFSET] != 0
    assert all(
        np.isclose(result.parameters[p], 0.0)
        for p in SurfaceTerms
        if p != SurfaceTerms.OFFSET
    )


@pytest.mark.integration
def test_map_level_reference_point_has_no_effect_with_none(
    scan_image_with_nans: ScanImage,
):
    result_centered = level_map(scan_image_with_nans, SurfaceTerms.NONE)
    result_ref = level_map(
        scan_image_with_nans, SurfaceTerms.NONE, reference_point=(10.5, -5.2)
    )
    assert np.allclose(
        result_centered.leveled_map, result_ref.leveled_map, equal_nan=True
    )


@pytest.mark.integration
def test_map_level_reference_point_has_effect_with_plane(
    scan_image_with_nans: ScanImage,
):
    result_centered = level_map(scan_image_with_nans, SurfaceTerms.PLANE)
    result_ref = level_map(
        scan_image_with_nans, SurfaceTerms.NONE, reference_point=(10.5, -5.2)
    )
    assert not np.allclose(
        result_centered.leveled_map, result_ref.leveled_map, equal_nan=True
    )
