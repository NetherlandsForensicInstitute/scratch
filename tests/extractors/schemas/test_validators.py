from contextlib import chdir
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from extractors.schemas import RelativePath, SupportedExtension

relative_path_adapter = TypeAdapter(RelativePath)


@pytest.mark.usefixtures("tmp_dir_api")
class TestRelativePathType:
    """Tests for the RelativePath type alias with extension validation."""

    @pytest.fixture(autouse=True)
    def _change_to_tmp_path(self, tmp_path: Path):
        """Change to tmp_path directory for all tests in this class."""
        with chdir(tmp_path):
            yield

    @pytest.mark.parametrize(
        "filename",
        [pytest.param(f"test.{ext.value}", id=f"extension_{ext.value}") for ext in SupportedExtension]
        + [
            pytest.param("file.name.with.dots.png", id="multiple_dots"),
            pytest.param("./test.png", id="reference current directory"),
            pytest.param("../test.png", id="reference parrent directory"),
            # pytest.param("sub-dir/test.png", id="sub directory"),
        ],
    )
    def test_valid_filenames(self, filename: str) -> None:
        """Test that valid filenames and extensions pass validation."""
        # Arrange
        test_file = Path.cwd() / filename
        test_file.write_text("test data")

        # Act
        result = relative_path_adapter.validate_python(filename)

        # Assert
        assert isinstance(result, Path)

    @given(
        extension=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=1,
            max_size=3,
        ).filter(lambda ext: ext not in [e.value for e in SupportedExtension])
    )
    def test_invalid_extensions_fail(self, extension: str) -> None:
        """Test that random invalid extensions fail validation using property-based testing."""
        # Act & Assert
        with pytest.raises(ValidationError, match="unsupported file type"):
            relative_path_adapter.validate_python(f"test.{extension}")

    def test_empty_string_fails(self) -> None:
        """Test that empty string fails validation."""
        # Act & Assert
        with pytest.raises(ValidationError, match="unsupported file type"):
            relative_path_adapter.validate_python("")

    def test_absolute_paths_fail(self) -> None:
        """Test that absolute paths fail validation."""
        # Act & Assert
        with pytest.raises(ValidationError, match="absolute paths are not allowed"):
            relative_path_adapter.validate_python("/")
