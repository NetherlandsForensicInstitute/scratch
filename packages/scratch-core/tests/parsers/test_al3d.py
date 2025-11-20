"""Tests for AL3D file loader."""

from pathlib import Path

import numpy as np
import pytest

from models.enums import ImageType, SupportedExtension
from models.image import ImageData
from parsers.al3d import load_al3d_file
from parsers.extract_al3d_resolution import _extract_resolution_from_description


def is_git_lfs_pointer(file_path: Path) -> bool:
    """Check if file is a Git LFS pointer (not the actual file)."""
    try:
        with open(file_path, "rb") as f:
            first_line = f.read(100)
        return b"version https://git-lfs.github.com" in first_line
    except (OSError, IOError):
        return False


@pytest.fixture(scope="module")
def circle_al3d(scans_dir: Path) -> ImageData:
    return load_al3d_file(scans_dir / "circle.al3d")


def test_load_al3d_file_has_dimensions(circle_al3d: ImageData) -> None:
    """Test that loaded AL3D file has valid dimensions."""
    # Dimension checks (AL3D files are always surfaces)
    # NOTE: Are we testing our code or surfalize
    assert 0 < circle_al3d.xdim < 1, "xdim should be between 0 and 1 meter"
    assert 0 < circle_al3d.ydim < 1, "ydim should be between 0 and 1 meter"
    assert 0.1 < circle_al3d.xdim / circle_al3d.ydim < 10, (
        "xdim and ydim should be similar scale"
    )
    assert circle_al3d.xdim_orig == circle_al3d.xdim
    assert circle_al3d.ydim_orig == circle_al3d.ydim


def test_load_al3d_file_has_metadata(circle_al3d: ImageData) -> None:
    """Test that loaded AL3D file contains metadata."""

    # Metadata checks
    additional_info = circle_al3d.additional_info
    assert isinstance(additional_info["Header"], dict)
    assert isinstance(additional_info["XMLData"], dict)


def test_load_al3d_invalid_pixels_are_nan(circle_al3d: ImageData) -> None:
    """Test that invalid pixels (> 1e9) are converted to NaN."""

    # Check that there are no values > 1e9 (they should be NaN)
    assert np.all(circle_al3d.depth_data[~np.isnan(circle_al3d.depth_data)] < 1e9), (
        "All non-NaN values should be < 1e9"
    )

    # Check that invalid_pixel_val is set to NaN
    assert np.isnan(circle_al3d.invalid_pixel_val)


def test_load_al3d_surface_data_shape(circle_al3d: ImageData) -> None:
    """Test that surface data has correct shape (2D)."""

    depth_data = circle_al3d.depth_data

    # AL3D files always contain surface data
    assert depth_data.ndim == 2, "Surface data should be 2D"
    assert depth_data.shape[0] > 0
    assert depth_data.shape[1] > 0
    assert depth_data.dtype == np.float64, "Depth data should be float64"


def test_load_all_al3d_files(al3d_file_path: Path) -> None:
    """Test that all AL3D files in resources can be loaded."""

    # Skip if this is a Git LFS pointer (file not actually downloaded)
    if is_git_lfs_pointer(al3d_file_path) or al3d_file_path.stem == "circle":
        pytest.skip(
            f"{al3d_file_path} is a Git LFS pointer - actual file not downloaded"
        )

    result = load_al3d_file(al3d_file_path)

    # Basic validation
    assert isinstance(result, ImageData)
    assert result.type == ImageType.SURFACE
    assert result.input_format == SupportedExtension.AL3D
    assert result.depth_data is not None
    assert result.depth_data.size > 0, f"File {al3d_file_path} has empty depth data"
    assert result.xdim > 0, f"File {al3d_file_path} has invalid xdim"
    assert result.ydim > 0, f"File {al3d_file_path} has invalid ydim"
    assert result.orig_path == str(al3d_file_path)


def test_load_al3d_file_not_found() -> None:
    """Test that loading non-existent file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        load_al3d_file(Path("/nonexistent/file.al3d"))


def test_load_al3d_preprocessing_flags(circle_al3d: ImageData) -> None:
    """Test that preprocessing flags are set correctly."""

    # Newly loaded files should not have any preprocessing flags set
    assert circle_al3d.is_prep is False
    assert circle_al3d.is_crop is False
    assert circle_al3d.is_interp is False
    assert circle_al3d.is_resamp is False


def test_load_al3d_default_values(circle_al3d: ImageData) -> None:
    """Test that default values are set correctly."""

    # Default values from MATLAB translation
    assert circle_al3d.mark_type == ""
    assert circle_al3d.subsampling == 1
    assert circle_al3d.crop_info == []
    assert circle_al3d.cutoff_hi == []
    assert circle_al3d.cutoff_lo == []
    assert circle_al3d.data_param == {}


@pytest.mark.parametrize(
    "description",
    (
        pytest.param(
            "Some text before\nEstimated Vertical Resolution: 2.5 µm\nSome text after",
            id="line feed",
        ),
        pytest.param(
            "Some text before\rEstimated Vertical Resolution: 2.5 µm\rSome text after",
            id="carriage return",
        ),
        pytest.param(
            "Some text before\r\nEstimated Vertical Resolution: 2.5 µm\r\nSome text after",
            id="crlf",
        ),
        pytest.param(
            "Estimated Vertical Resolution:     2.5 µm",
            id="whitespace",
        ),
    ),
)
def test_extract_resolution_from_description(description: str) -> None:
    """Test extraction of resolution from XML description."""

    result = _extract_resolution_from_description(description, "Vertical")

    assert result and np.isclose(result, 2.5e-6)


@pytest.mark.parametrize(
    "description",
    (
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("Estimated Lateral Resolution: 0.5", id="no unit"),
        pytest.param("Estimated Lateral Resolution: mm", id="no value"),
        pytest.param("some prefix: 0.5 mm", id="bad prefix"),
        pytest.param("0.5 mm", id="no prefix"),
    ),
)
def test_extract_resolution_none_value(description: str) -> None:
    """Test that None is returned when resolution is not found."""

    assert _extract_resolution_from_description(description, "Vertical") is None


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
