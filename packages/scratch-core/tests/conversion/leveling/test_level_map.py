from conversion.leveling import level_map, SurfaceTerms
from image_generation.data_formats import ScanImage


def test_map_level_sphere(scan_image_with_nans: ScanImage):
    result = level_map(scan_image_with_nans, SurfaceTerms.SPHERE)
    assert result


def test_map_level_plane(scan_image_with_nans: ScanImage):
    result = level_map(scan_image_with_nans, SurfaceTerms.PLANE)
    assert result


def test_map_level_none(scan_image_with_nans: ScanImage):
    result = level_map(scan_image_with_nans, SurfaceTerms.NONE)
    assert result
