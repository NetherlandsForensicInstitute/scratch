"""Tests for AL3D file loader."""

from pathlib import Path
from typing import Final

import numpy as np
import pytest

from models.enums import ImageType, SupportedExtension
from models.image import ImageData
from parsers.al3d import load_al3d_file
from parsers.extract_al3d_resolution import _extract_resolution_from_description

AL3D_FILES: Final[tuple[str, ...]] = (
    "circle.al3d",
    "Huls1.al3d",
    "Huls2.al3d",
    "Huls3.al3d",
    "Huls4.al3d",
    "Klein_replica_mode.al3d",
    "Klein_non_replica_mode.al3d",
)


def is_git_lfs_pointer(file_path: Path) -> bool:
    """Check if file is a Git LFS pointer (not the actual file)."""
    try:
        with open(file_path, "rb") as f:
            first_line = f.read(100)
        return b"version https://git-lfs.github.com" in first_line
    except (OSError, IOError):
        return False


def test_load_al3d_file_basic(scans_dir: Path) -> None:
    """Test that we can load a basic AL3D file without errors."""
    al3d_file = scans_dir / "circle.al3d"

    # Skip if this is a Git LFS pointer
    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Basic structural checks
    assert isinstance(result, ImageData)
    assert result.type == ImageType.SURFACE
    assert isinstance(result.depth_data, np.ndarray)


def test_load_al3d_file_has_dimensions(scans_dir: Path) -> None:
    """Test that loaded AL3D file has valid dimensions."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Dimension checks (AL3D files are always surfaces)
    assert result.xdim > 0, "xdim should be positive"
    assert result.ydim > 0, "ydim should be positive for surface data"


def test_load_al3d_file_has_metadata(scans_dir: Path) -> None:
    """Test that loaded AL3D file contains metadata."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Metadata checks
    assert result.additional_info is not None
    assert "Header" in result.additional_info
    assert isinstance(result.additional_info["Header"], dict)


def test_load_al3d_file_has_correct_format(scans_dir: Path) -> None:
    """Test that AL3D file has correct input_format."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Format check
    assert result.input_format == SupportedExtension.AL3D


def test_load_al3d_invalid_pixels_are_nan(scans_dir: Path) -> None:
    """Test that invalid pixels (> 1e9) are converted to NaN."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Check that there are no values > 1e9 (they should be NaN)
    valid_data = result.depth_data[~np.isnan(result.depth_data)]
    assert np.all(valid_data < 1e9), "All non-NaN values should be < 1e9"

    # Check that invalid_pixel_val is set to NaN
    assert np.isnan(result.invalid_pixel_val)


def test_load_all_al3d_files(al3d_file_path: Path) -> None:
    """Test that all AL3D files in resources can be loaded."""

    # Skip if this is a Git LFS pointer (file not actually downloaded)
    if is_git_lfs_pointer(al3d_file_path):
        pytest.skip(
            f"{al3d_file_path} is a Git LFS pointer - actual file not downloaded"
        )

    result = load_al3d_file(al3d_file_path)

    # Basic validation
    assert isinstance(result, ImageData)
    assert result.type == ImageType.SURFACE
    assert result.depth_data is not None
    assert result.depth_data.size > 0, f"File {al3d_file_path} has empty depth data"
    assert result.xdim > 0, f"File {al3d_file_path} has invalid xdim"
    assert result.ydim > 0, f"File {al3d_file_path} has invalid ydim"


def test_load_al3d_surface_data_shape(scans_dir: Path) -> None:
    """Test that surface data has correct shape (2D)."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # AL3D files always contain surface data
    assert result.depth_data.ndim == 2, "Surface data should be 2D"
    assert result.depth_data.shape[0] > 0
    assert result.depth_data.shape[1] > 0


def test_load_al3d_file_not_found() -> None:
    """Test that loading non-existent file raises appropriate error."""
    with pytest.raises(Exception):  # Could be FileNotFoundError or similar
        load_al3d_file(Path("/nonexistent/file.al3d"))


def test_load_al3d_dimensions_match_data(scans_dir: Path) -> None:
    """Test that xdim/ydim are reasonable given the data dimensions."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Dimensions should be in meters and positive
    assert 0 < result.xdim < 1, "xdim should be reasonable (between 0 and 1 meter)"
    assert 0 < result.ydim < 1, "ydim should be reasonable (between 0 and 1 meter)"

    # For surface data, dimensions should be relatively similar (not orders of magnitude apart)
    ratio = result.xdim / result.ydim
    assert 0.1 < ratio < 10, "xdim and ydim should be similar scale"


def test_load_al3d_depth_data_dtype(scans_dir: Path) -> None:
    """Test that depth data has appropriate dtype."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Depth data should be float64 for measurements
    assert result.depth_data.dtype == np.float64, "Depth data should be float64"


def test_load_al3d_orig_path_set(scans_dir: Path) -> None:
    """Test that orig_path is set correctly."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # orig_path should be set to the file path
    assert result.orig_path == str(al3d_file)


def test_load_al3d_preprocessing_flags(scans_dir: Path) -> None:
    """Test that preprocessing flags are set correctly."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Newly loaded files should not have any preprocessing flags set
    assert result.is_prep is False
    assert result.is_crop is False
    assert result.is_interp is False
    assert result.is_resamp is False


def test_load_al3d_default_values(scans_dir: Path) -> None:
    """Test that default values are set correctly."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # Default values from MATLAB translation
    assert result.mark_type == ""
    assert result.subsampling == 1
    assert result.crop_info == []
    assert result.cutoff_hi == []
    assert result.cutoff_lo == []
    assert result.data_param == {}


def test_extract_resolution_from_description_vertical() -> None:
    """Test extraction of vertical resolution from XML description."""
    # Sample description text with vertical resolution
    description = (
        "Some text before\nEstimated Vertical Resolution: 2.5 µm\nSome text after"
    )

    result = _extract_resolution_from_description(description, "Vertical")

    # 2.5 µm = 2.5e-6 m
    assert result is not None
    assert np.isclose(result, 2.5e-6)


def test_extract_resolution_from_description_lateral() -> None:
    """Test extraction of lateral resolution from XML description."""
    # Sample description text with lateral resolution
    description = (
        "Some text before\nEstimated Lateral Resolution: 1.0 µm\nSome text after"
    )

    result = _extract_resolution_from_description(description, "Lateral")

    # 1.0 µm = 1.0e-6 m
    assert result is not None
    assert np.isclose(result, 1.0e-6)


def test_extract_resolution_from_description_nanometers() -> None:
    """Test extraction with nanometer units."""
    description = "Estimated Vertical Resolution: 500 nm"

    result = _extract_resolution_from_description(description, "Vertical")

    # 500 nm = 500e-9 m
    assert result is not None
    assert np.isclose(result, 500e-9)


def test_extract_resolution_from_description_millimeters() -> None:
    """Test extraction with millimeter units."""
    description = "Estimated Lateral Resolution: 0.5 mm"

    result = _extract_resolution_from_description(description, "Lateral")

    # 0.5 mm = 0.5e-3 m
    assert result is not None
    assert np.isclose(result, 0.5e-3)


def test_extract_resolution_from_description_meters() -> None:
    """Test extraction with meter units."""
    description = "Estimated Vertical Resolution: 0.001 m"

    result = _extract_resolution_from_description(description, "Vertical")

    # 0.001 m = 0.001 m
    assert result is not None
    assert np.isclose(result, 0.001)


def test_extract_resolution_from_description_not_found() -> None:
    """Test that None is returned when resolution is not found."""
    description = "Some random text without resolution info"

    result = _extract_resolution_from_description(description, "Vertical")

    assert result is None


def test_extract_resolution_from_description_empty() -> None:
    """Test that None is returned for empty description."""
    result = _extract_resolution_from_description("", "Vertical")
    assert result is None


def test_extract_resolution_from_description_whitespace_handling() -> None:
    """Test that whitespace is handled correctly (matching MATLAB behavior)."""
    # Description with extra whitespace after colon
    description = "Estimated Vertical Resolution:     3.14 µm"

    result = _extract_resolution_from_description(description, "Vertical")

    assert result is not None
    assert np.isclose(result, 3.14e-6)


def test_extract_resolution_from_description_line_endings() -> None:
    """Test handling of different line endings (CR/LF)."""
    # Test with CR (char code 13)
    description_cr = "Estimated Vertical Resolution: 1.5 µm\rNext line"
    result_cr = _extract_resolution_from_description(description_cr, "Vertical")
    assert result_cr is not None
    assert np.isclose(result_cr, 1.5e-6)

    # Test with LF (char code 10)
    description_lf = "Estimated Vertical Resolution: 1.5 µm\nNext line"
    result_lf = _extract_resolution_from_description(description_lf, "Vertical")
    assert result_lf is not None
    assert np.isclose(result_lf, 1.5e-6)

    # Test with CRLF
    description_crlf = "Estimated Vertical Resolution: 1.5 µm\r\nNext line"
    result_crlf = _extract_resolution_from_description(description_crlf, "Vertical")
    assert result_crlf is not None
    assert np.isclose(result_crlf, 1.5e-6)


@pytest.mark.parametrize(
    "value,unit,expected",
    [
        ("1.0", "m", 1.0),
        ("1.0", "mm", 1.0e-3),
        ("1.0", "µm", 1.0e-6),
        ("1.0", "nm", 1.0e-9),
        ("2.5", "µm", 2.5e-6),
        ("100", "nm", 100e-9),
        ("0.5", "mm", 0.5e-3),
    ],
)
def test_extract_resolution_unit_conversion(
    value: str, unit: str, expected: float
) -> None:
    """Test unit conversion for different unit types."""
    description = f"Estimated Vertical Resolution: {value} {unit}"

    result = _extract_resolution_from_description(description, "Vertical")

    assert result is not None
    assert np.isclose(result, expected)


def test_load_al3d_xdim_orig_equals_xdim(scans_dir: Path) -> None:
    """Test that xdim_orig equals xdim for newly loaded files."""
    al3d_file = scans_dir / "circle.al3d"

    if is_git_lfs_pointer(al3d_file):
        pytest.skip("circle.al3d is a Git LFS pointer - actual file not downloaded")

    result = load_al3d_file(al3d_file)

    # For newly loaded files, orig dimensions should match current dimensions
    assert result.xdim_orig == result.xdim
    assert result.ydim_orig == result.ydim
