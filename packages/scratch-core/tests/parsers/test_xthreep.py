"""Tests for X3P file loader."""

from typing import Final
import zipfile
from pathlib import Path

import numpy as np
import pytest

from models.enums import ImageType
from models.image import ImageData
from parsers.xthreep import load_x3p_file

X3P_FILES: Final[tuple[str, ...]] = (
    "circle.x3p",
    "Huls1_X3P_Scratch.x3p",
    "Huls2_X3P_Scratch.x3p",
    "Huls3_X3P_Scratch.x3p",
    "Huls4_X3P_Scratch.x3p",
    "Klein_replica_mode_X3P_Scratch.x3p",
    "Klein_replica_mode_X3P_Alicona.x3p",
    "Klein_non_replica_mode_X3P_Scratch.x3p",
)


def is_git_lfs_pointer(file_path: Path) -> bool:
    """Check if file is a Git LFS pointer (not the actual file)."""
    try:
        with open(file_path, "r") as f:
            first_line = f.readline()
        return first_line.startswith("version https://git-lfs.github.com")
    except (UnicodeDecodeError, OSError):
        return False


def is_valid_x3p_file(file_path: Path) -> bool:
    """Check if file is a valid X3P file (zip archive)."""
    if is_git_lfs_pointer(file_path):
        return False
    try:
        with zipfile.ZipFile(file_path, "r") as _:
            return True
    except (zipfile.BadZipFile, OSError):
        return False


def test_load_x3p_file_basic(scans_dir: Path) -> None:
    """Test that we can load a basic X3P file without errors."""
    x3p_file = scans_dir / "circle.x3p"
    result = load_x3p_file(x3p_file)

    # Basic structural checks
    assert isinstance(result, ImageData)
    assert result.type in [ImageType.SURFACE, ImageType.PROFILE]
    assert isinstance(result.depth_data, np.ndarray)


def test_load_x3p_file_has_dimensions(scans_dir: Path) -> None:
    """Test that loaded X3P file has valid dimensions."""
    x3p_file = scans_dir / "circle.x3p"
    result = load_x3p_file(x3p_file)

    # Dimension checks
    assert result.xdim > 0, "xdim should be positive"
    match result.type:
        case ImageType.SURFACE:
            assert result.ydim > 0, "ydim should be positive for surface data"
        case ImageType.PROFILE:
            assert not result.ydim, "ydim should be 0 for profile data"


def test_load_x3p_file_has_metadata(scans_dir: Path) -> None:
    """Test that loaded X3P file contains metadata."""
    x3p_file = scans_dir / "circle.x3p"
    result = load_x3p_file(x3p_file)

    # Metadata checks
    assert result.additional_info.get("pinfo")


@pytest.mark.parametrize("filename", X3P_FILES)
def test_load_all_x3p_files(filename: str, scans_dir: Path) -> None:
    """Test that all X3P files in resources can be loaded."""
    x3p_file = scans_dir / filename

    # Skip if this is a Git LFS pointer (file not actually downloaded)
    if is_git_lfs_pointer(x3p_file):
        pytest.skip(f"{filename} is a Git LFS pointer - actual file not downloaded")

    result = load_x3p_file(x3p_file)

    # Basic validation
    assert isinstance(result, ImageData)
    assert result.depth_data is not None
    assert result.depth_data.size > 0, f"File {filename} has empty depth data"
    assert result.xdim > 0, f"File {filename} has invalid xdim"


def test_load_x3p_surface_data_shape(scans_dir: Path) -> None:
    """Test that surface data has correct shape (2D)."""
    x3p_file = scans_dir / "circle.x3p"
    result = load_x3p_file(x3p_file)

    if result.type == ImageType.SURFACE:
        assert result.depth_data.ndim == 2, "Surface data should be 2D"
        assert result.depth_data.shape[0] > 0
        assert result.depth_data.shape[1] > 0


@pytest.mark.parametrize("filename", X3P_FILES)
def test_load_x3p_profile_data_shape(filename: str, scans_dir: Path) -> None:
    """Test that profile data has correct shape (1D)."""
    x3p_file = scans_dir / filename

    # Skip LFS pointers (files not actually downloaded)
    if is_git_lfs_pointer(x3p_file):
        pytest.skip(f"{filename} is a Git LFS pointer - actual file not downloaded")

    result = load_x3p_file(x3p_file)

    # Only assert shape if it's actually a profile
    if result.type == ImageType.PROFILE:
        assert result.depth_data.ndim == 1, f"{filename}: Profile data should be 1D"
        assert result.depth_data.shape[0] > 0


def test_load_x3p_file_not_found() -> None:
    """Test that loading non-existent file raises appropriate error."""
    with pytest.raises(Exception):  # Could be FileNotFoundError or similar
        load_x3p_file(Path("/nonexistent/file.x3p"))


def test_load_x3p_dimensions_match_data(scans_dir: Path) -> None:
    """Test that xdim/ydim are reasonable given the data dimensions."""
    x3p_file = scans_dir / "circle.x3p"
    result = load_x3p_file(x3p_file)

    # Dimensions should be in meters and positive
    assert 0 < result.xdim < 1, "xdim should be reasonable (between 0 and 1 meter)"

    if result.type == ImageType.SURFACE:
        assert 0 < result.ydim < 1, "ydim should be reasonable (between 0 and 1 meter)"
        # For surface data, dimensions should be relatively similar (not orders of magnitude apart)
        ratio = result.xdim / result.ydim
        assert 0.1 < ratio < 10, "xdim and ydim should be similar scale"


def test_load_x3p_depth_data_dtype(scans_dir: Path) -> None:
    """Test that depth data has appropriate dtype."""
    x3p_file = scans_dir / "circle.x3p"
    result = load_x3p_file(x3p_file)

    # Depth data should be float type for measurements
    assert np.issubdtype(result.depth_data.dtype, np.floating), (
        "Depth data should be floating point"
    )


def test_load_x3p_optional_fields(scans_dir: Path) -> None:
    """Test that optional fields are set correctly."""
    x3p_file = scans_dir / "circle.x3p"
    result = load_x3p_file(x3p_file)

    # These fields should be None or have default values for basic X3P files
    # (based on the MATLAB code translation)
    assert result.texture_data is None, "Basic X3P should not have texture data"
    assert result.quality_data is None, "Basic X3P should not have quality data"
    assert result.vertical_resolution is None
    assert result.lateral_resolution is None
