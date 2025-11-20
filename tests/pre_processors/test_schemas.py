from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pre_processors.schemas import SupportedExtension, UploadScan


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def valid_scan_file(tmp_path: Path) -> Path:
    """Create a temporary scan file with valid extension."""
    scan_file = tmp_path / "test_scan.x3p"
    scan_file.touch()
    return scan_file


@pytest.mark.parametrize(
    "extension",
    [ext.value for ext in SupportedExtension],
)
def test_all_supported_extensions(tmp_path: Path, output_dir: Path, extension: str) -> None:
    """Test that all supported extensions are accepted."""
    # Arrange
    scan_file = tmp_path / f"test_scan.{extension}"
    scan_file.touch()

    # Act
    upload_scan = UploadScan(scan_file=scan_file, output_dir=output_dir)

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
        UploadScan(scan_file=scan_file, output_dir=output_dir)
    assert "unsupported extension" in str(exc_info.value)


def test_nonexistent_scan_file_raises_error(output_dir: Path) -> None:
    """Test that non-existent scan file raises ValidationError."""
    # Arrange
    nonexistent_file = Path("/nonexistent/path/scan.mat")
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=nonexistent_file, output_dir=output_dir)
    assert "Path does not point to a file" in str(exc_info.value)


def test_nonexistent_output_dir_raises_error(valid_scan_file: Path) -> None:
    """Test that non-existent output directory raises ValidationError."""
    # Arrange
    nonexistent_dir = Path("/nonexistent/output/dir")
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=valid_scan_file, output_dir=nonexistent_dir)
    assert "Path does not point to a directory" in str(exc_info.value)


def test_scan_file_as_directory_raises_error(tmp_path: Path, output_dir: Path) -> None:
    """Test that providing a directory as scan_file raises ValidationError."""
    # Arrange
    directory = tmp_path / "not_a_file"
    directory.mkdir()
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=directory, output_dir=output_dir)
    assert "Path does not point to a file" in str(exc_info.value)


def test_output_dir_as_file_raises_error(tmp_path: Path, valid_scan_file: Path) -> None:
    """Test that providing a file as output_dir raises ValidationError."""
    # Arrange
    file_not_dir = tmp_path / "not_a_directory.txt"
    file_not_dir.touch()
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=valid_scan_file, output_dir=file_not_dir)
    assert "Path does not point to a directory" in str(exc_info.value)


def test_model_is_frozen(valid_scan_file: Path, output_dir: Path) -> None:
    """Test that UploadScan instances are immutable (frozen)."""
    # Arrange
    upload_scan = UploadScan(
        scan_file=valid_scan_file,
        output_dir=output_dir,
    )
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        upload_scan.scan_file = valid_scan_file
    assert "Instance is frozen" in str(exc_info.value)


def test_extra_fields_forbidden(valid_scan_file: Path, output_dir: Path) -> None:
    """Test that extra fields are not allowed."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(
            scan_file=valid_scan_file,
            output_dir=output_dir,
            extra_field="not allowed",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
