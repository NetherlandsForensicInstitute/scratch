from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from preprocessors.schemas import SupportedExtension, UploadScan


@pytest.fixture
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary output directory."""
    return tmp_path_factory.mktemp("upload_scan_schema")


@pytest.fixture(scope="module")
def valid_scan_file(scan_directory: Path) -> Path:
    return scan_directory / "circle.x3p"


@pytest.mark.parametrize(
    "extension",
    [ext.value for ext in SupportedExtension],
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
    ).filter(lambda ext: ext not in [e.value for e in SupportedExtension])
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
    assert "try: al3d, x3p, sur, plu" in error_message


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


def test_tag_defaults_to_stem_when_project_name_not_provided(tmp_path: Path) -> None:
    """Test that tag property defaults to scan file stem when project_name is not provided."""
    # Arrange
    scan_file = tmp_path / "my_scan_file.x3p"
    scan_file.write_text("content")

    # Act
    upload_scan = UploadScan(scan_file=scan_file)  # type: ignore

    # Assert
    assert upload_scan.tag == "my_scan_file"


def test_tag_uses_project_name_when_provided(tmp_path: Path) -> None:
    """Test that tag property uses project_name when it is provided."""
    # Arrange
    scan_file = tmp_path / "scan_file.x3p"
    scan_file.write_text("content")

    # Act
    upload_scan = UploadScan(scan_file=scan_file, project_name="custom-project")  # type: ignore

    # Assert
    assert upload_scan.tag == "custom-project"
