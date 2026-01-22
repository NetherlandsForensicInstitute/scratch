from collections.abc import Callable
from pathlib import Path

import pytest
from container_models.light_source import LightSource
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from models import SupportedScanExtension
from preprocessors.schemas import UploadScan


@pytest.fixture
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary output directory."""
    return tmp_path_factory.mktemp("upload_scan_schema")


@pytest.fixture(scope="module")
def valid_scan_file(scan_directory: Path) -> Path:
    return scan_directory / "circle.x3p"


@pytest.mark.parametrize(
    "extension",
    [ext.value for ext in SupportedScanExtension],
)
def test_all_supported_extensions(tmp_path: Path, extension: str) -> None:
    """Test that all supported extensions are accepted."""
    # Arrange
    scan_file = tmp_path / f"test_scan.{extension}"
    scan_file.write_text("just words")

    # Act
    upload_scan = UploadScan(scan_file=scan_file)  # type: ignore

    # Assert
    assert upload_scan.scan_file == scan_file


@given(
    extension=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        min_size=1,
        max_size=10,
    ).filter(lambda ext: ext not in [e.value for e in SupportedScanExtension])
)
def test_unsupported_extension_raises_error(extension: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test that unsupported file extensions raise ValueError using property-based testing."""
    # Arrange
    tmp_path = tmp_path_factory.mktemp("test_bad_extensions")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    scan_file = tmp_path / f"test_scan.{extension}"
    scan_file.touch()

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=scan_file)  # type: ignore
    error_message = str(exc_info.value)
    assert "unsupported file type" in error_message
    assert "try: al3d, x3p" in error_message


def test_nonexistent_scan_file_raises_error() -> None:
    """Test that non-existent scan file raises ValidationError."""
    # Arrange
    nonexistent_file = Path("/nonexistent/path/scan.mat")

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=nonexistent_file)  # type: ignore
    assert "Path does not point to a file" in str(exc_info.value)


def test_scan_file_as_directory_raises_error(tmp_path: Path) -> None:
    """Test that providing a directory as scan_file raises ValidationError."""
    # Arrange
    directory = tmp_path / "not_a_file"
    directory.mkdir()

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=directory)  # type: ignore
    assert "Path does not point to a file" in str(exc_info.value)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"#!/bin/bash\necho 'malicious script'", id="shebang"),
        pytest.param(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 100, id="elf"),
        pytest.param(b"MZ" + b"\x00" * 100, id="pe"),
    ],
)
def test_executable_files_rejected(tmp_path: Path, content: bytes) -> None:
    """Test that executable files are rejected."""
    # Arrange
    executable_file = tmp_path / "test.x3p"
    executable_file.write_bytes(content)

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=executable_file)  # type: ignore
    assert "executable files are not allowed" in str(exc_info.value)


def test_tag_defaults_to_stem_when_project_name_not_provided(upload_scan_parameter: Callable[..., UploadScan]) -> None:
    """Test that tag property defaults to scan file stem when project_name is not provided."""
    # Act
    upload_scan = upload_scan_parameter()

    # Assert
    assert upload_scan.tag == upload_scan.scan_file.stem


def test_tag_uses_project_name_when_provided(upload_scan_parameter: Callable[..., UploadScan]) -> None:
    """Test that tag property uses project_name when it is provided."""
    # Arrange
    project_name = "custom-project"
    # Act
    upload_scan = upload_scan_parameter(project_name=project_name)

    # Assert
    assert upload_scan.tag == project_name


def test_default_values(upload_scan: UploadScan) -> None:
    """Test that default parameters are set correctly."""
    # Assert
    assert upload_scan.light_sources == (
        LightSource(azimuth=90, elevation=45),
        LightSource(azimuth=180, elevation=45),
    )
    assert upload_scan.observer == LightSource(azimuth=90, elevation=45)
    assert upload_scan.scale_x == 1.0
    assert upload_scan.scale_y == 1.0
    assert upload_scan.step_size_x == 1
    assert upload_scan.step_size_y == 1


def test_custom_parameters(upload_scan_parameter: Callable[..., UploadScan]) -> None:
    """Test that custom parameters can be set."""
    # Arrange
    custom_light = LightSource(azimuth=45, elevation=30)
    custom_observer = LightSource(azimuth=0, elevation=90)

    # Act
    params = upload_scan_parameter(  # type: ignore
        light_sources=(custom_light,),
        observer=custom_observer,
        scale_x=2.5,
        scale_y=3.0,
        step_size_x=2,
        step_size_y=3,
    )

    # Assert
    assert params.light_sources == (custom_light,)
    assert params.observer == custom_observer
    assert params.scale_x == 2.5  # noqa: PLR2004
    assert params.scale_y == 3.0  # noqa: PLR2004
    assert params.step_size_x == 2  # noqa: PLR2004
    assert params.step_size_y == 3  # noqa: PLR2004


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("scale_x", 0.0),
        ("scale_x", -1.0),
        ("scale_y", 0.0),
        ("scale_y", -1.5),
        ("step_size_x", 0),
        ("step_size_x", -1),
        ("step_size_y", 0),
        ("step_size_y", -2),
    ],
)
def test_invalid_scale_and_step_values(
    field_name: str, invalid_value: float | int, upload_scan_parameter: Callable[..., UploadScan]
) -> None:
    """Test that scale and step size values must be positive."""
    # Arrange
    valid_params = {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "step_size_x": 1,
        "step_size_y": 1,
    }
    valid_params[field_name] = invalid_value

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:  # Pydantic raises ValidationError
        upload_scan_parameter(**valid_params)  # type: ignore

    # Verify the error is related to the constraint
    assert "greater than" in str(exc_info.value).lower()
