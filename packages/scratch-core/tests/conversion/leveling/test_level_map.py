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
def test_map_level_none_has_no_effect(scan_image_with_nans: ScanImage):
    result = level_map(scan_image_with_nans, SurfaceTerms.NONE)
    assert result
    assert np.allclose(result.leveled_map, scan_image_with_nans.data, equal_nan=True)


@pytest.mark.integration
@pytest.mark.parametrize(
    "terms",
    [
        SurfaceTerms.OFFSET,
        SurfaceTerms.ASTIG_0,
        SurfaceTerms.ASTIG_45,
        SurfaceTerms.DEFOCUS,
        SurfaceTerms.TILT_X,
        SurfaceTerms.TILT_Y,
        SurfaceTerms.TILT_X | SurfaceTerms.TILT_Y,
        SurfaceTerms.OFFSET | SurfaceTerms.ASTIG_0 | SurfaceTerms.ASTIG_45,
    ],
)
def test_map_level_raises_on_incorrect_term(
    scan_image_with_nans: ScanImage, terms: SurfaceTerms
):
    with pytest.raises(ValueError, match="No degree defined for"):
        _ = level_map(scan_image_with_nans, terms)
