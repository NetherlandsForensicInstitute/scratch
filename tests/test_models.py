from enum import StrEnum, auto
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from extractors.schemas import ProcessedDataAccess
from models import (
    BaseModelConfig,
    DirectoryAccess,
    validate_file_extension,
    validate_not_executable,
    validate_relative_path,
)
from preprocessors.schemas import UploadScan, UploadScanParameters
from settings import get_settings


class SampleExtensions(StrEnum):
    """Sample enum for file extensions used in tests."""

    X3P = auto()
    PNG = auto()
    TXT = auto()


@given(
    path=st.one_of(
        st.just(""),  # No path
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=10,
        ).map(lambda x: f"/{x}/"),  # Absolute path
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=10,
        ).map(lambda x: f"{x}/"),  # Relative path
    ),
    basename=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
        min_size=1,
        max_size=20,
    ),
    sub_extension=st.one_of(
        st.just(""),  # No sub-extension
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=10,
        ).map(lambda x: f".{x}"),  # Sub-extension like .tar, .backup, etc.
    ),
    extension=st.sampled_from(SampleExtensions),
)
def test_validate_file_extension_valid(path: str, basename: str, sub_extension: str, extension: str) -> None:
    """Test that valid file extensions pass validation with various paths using property-based testing."""
    # Arrange
    filename = Path(path) / f"{basename}{sub_extension}.{extension}"

    # Act
    result = validate_file_extension(filename, SampleExtensions)

    # Assert
    assert result == filename


@given(
    invalid_filename=st.one_of(
        # Empty string
        st.just(""),
        # Files with no extension
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ),
        # Files with invalid extensions
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).flatmap(
            lambda basename: st.sampled_from([".pdf", ".jpg", ".docx", ".zip", ".tar"]).map(
                lambda ext: f"{basename}{ext}"
            )
        ),
        # Files with wrong case extensions
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).flatmap(
            lambda basename: st.sampled_from([".X3P", ".PNG", ".TXT", ".X3p", ".Png"]).map(
                lambda ext: f"{basename}{ext}"
            )
        ),
        # Partial extension matches (extension + extra characters)
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).flatmap(
            lambda basename: st.sampled_from(SampleExtensions).flatmap(
                lambda ext: st.text(
                    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
                    min_size=1,
                    max_size=3,
                ).map(lambda suffix: f"{basename}{ext}{suffix}")
            )
        ),
    )
)
def test_validate_file_extension_invalid(invalid_filename: str) -> None:
    """Test that invalid filenames raise ValueError using property-based testing."""
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        validate_file_extension(Path(invalid_filename), SampleExtensions)

    # Assert
    error_message = str(exc_info.value)
    assert "unsupported file type" in error_message
    assert "try: x3p, png, txt" in error_message


def test_validate_not_executable_with_shebang(tmp_path: Path) -> None:
    """Test that script files with shebang are rejected."""
    # Arrange
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        validate_not_executable(script_file)
    assert "executable files are not allowed" in str(exc_info.value)


def test_validate_not_executable_with_elf_binary(tmp_path: Path) -> None:
    """Test that ELF binary executables are rejected."""
    # Arrange
    elf_file = tmp_path / "test_binary"
    elf_file.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 100)

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        validate_not_executable(elf_file)
    assert "executable files are not allowed" in str(exc_info.value)


def test_validate_not_executable_with_pe_executable(tmp_path: Path) -> None:
    """Test that Windows PE executables are rejected."""
    # Arrange
    pe_file = tmp_path / "test.exe"
    pe_file.write_bytes(b"MZ" + b"\x00" * 100)

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        validate_not_executable(pe_file)
    assert "executable files are not allowed" in str(exc_info.value)


@pytest.mark.parametrize(
    "mach_o_header",
    [
        b"\xfe\xed\xfa\xce",  # Mach-O 32-bit
        b"\xfe\xed\xfa\xcf",  # Mach-O 64-bit
        b"\xce\xfa\xed\xfe",  # Mach-O 32-bit reverse
        b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit reverse
    ],
)
def test_validate_not_executable_with_mach_o_binary(tmp_path: Path, mach_o_header: bytes) -> None:
    """Test that Mach-O binary executables are rejected."""
    # Arrange
    mach_o_file = tmp_path / "test_binary"
    mach_o_file.write_bytes(mach_o_header + b"\x00" * 100)

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        validate_not_executable(mach_o_file)
    assert "executable files are not allowed" in str(exc_info.value)


def test_validate_not_executable_with_valid_file(tmp_path: Path) -> None:
    """Test that non-executable files pass validation."""
    # Arrange
    valid_file = tmp_path / "data.txt"
    valid_file.write_text("just some data\nno executable content here")

    # Act
    result = validate_not_executable(valid_file)

    # Assert
    assert result == valid_file


def test_validate_not_executable_with_binary_data(tmp_path: Path) -> None:
    """Test that binary files without executable magic bytes pass validation."""
    # Arrange
    binary_file = tmp_path / "data.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03" + b"\xff" * 100)

    # Act
    result = validate_not_executable(binary_file)

    # Assert
    assert result == binary_file


def test_validate_relative_path_with_absolute_path() -> None:
    """Test that absolute paths are rejected."""
    # Arrange
    absolute_path = Path("/etc/passwd")

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        validate_relative_path(absolute_path)
    assert "absolute paths are not allowed" in str(exc_info.value)


def test_validate_relative_path_with_relative_path() -> None:
    """Test that relative paths pass validation."""
    # Arrange
    relative_path = Path("relative/path/file.txt")

    # Act
    result = validate_relative_path(relative_path)

    # Assert
    assert result == relative_path


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("file.txt"),
        Path("subdir/file.txt"),
        Path("./file.txt"),
        Path("../parent/file.txt"),
        Path("deeply/nested/path/to/file.txt"),
    ],
)
def test_validate_relative_path_with_various_relative_paths(relative_path: Path) -> None:
    """Test that various relative path formats pass validation."""
    # Act
    result = validate_relative_path(relative_path)

    # Assert
    assert result == relative_path


@pytest.mark.parametrize(
    "absolute_path",
    [
        Path("/"),
        Path("/etc/passwd"),
        Path("/usr/bin/python"),
        Path("/home/user/file.txt"),
        Path.cwd() / "file.txt",
    ],
)
def test_validate_relative_path_with_various_absolute_paths(absolute_path: Path) -> None:
    """Test that various absolute path formats are rejected."""
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        validate_relative_path(absolute_path)
    assert "absolute paths are not allowed" in str(exc_info.value)


@pytest.mark.usefixtures("tmp_dir_api")
class TestDirectoryAccess:
    """Tests for DirectoryAccess model."""

    def test_unique_token_generation(self) -> None:
        """Test that tokens are unique and don't conflict with existing directories."""
        # Arrange
        first_access = DirectoryAccess(tag="test")

        # Create the directory to simulate it exists
        first_access.resource_path.mkdir(parents=True)

        # Act
        second_access = DirectoryAccess(tag="test")

        # Assert - tokens should be different
        assert first_access.token != second_access.token
        assert first_access.resource_path != second_access.resource_path

    def test_resource_path_format(self) -> None:
        """Test that resource_path follows the expected format."""
        # Act
        access = DirectoryAccess(tag="my-tag")

        # Assert
        assert access.resource_path.name == f"my-tag-{access.token.hex}"
        assert access.resource_path.parent == get_settings().storage


@pytest.mark.parametrize(
    "schema",
    [
        DirectoryAccess,
        ProcessedDataAccess,
        UploadScan,
        UploadScanParameters,
    ],
)
def test_schema_is_base_model_config(schema: type[BaseModelConfig]):
    assert issubclass(schema, BaseModelConfig)
