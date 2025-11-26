from pathlib import Path

import pytest
from pydantic import ValidationError

from preprocessors import ProcessedDataLocation


@pytest.fixture(scope="module")
def process_scan_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create temporary image files in the same directory."""
    scan_dir = tmp_path_factory.mktemp("scans")

    def create_dummy(file: str) -> Path:
        path = scan_dir / file
        path.touch()
        return path

    return {
        file_path.stem: file_path
        for file_path in map(create_dummy, ("x3p_image.x3p", "preview_image.png", "surfacemap_image.png"))
    }


def test_process_scan_valid_same_directory(process_scan_files: dict[str, Path]) -> None:
    """Test that ProcessedDataLocation accepts files from the same parent directory."""
    # Act
    process_scan = ProcessedDataLocation(**process_scan_files)

    # Assert
    assert process_scan.x3p_image == process_scan_files["x3p_image"]
    assert process_scan.preview_image == process_scan_files["preview_image"]
    assert process_scan.surfacemap_image == process_scan_files["surfacemap_image"]


@pytest.mark.parametrize("field", ["x3p_image", "preview_image", "surfacemap_image"])
def test_process_scan_nonexistent_file_raises_error(field: str, process_scan_files: dict[str, Path]) -> None:
    """Test that ProcessedDataLocation raises error for non-existent files."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ProcessedDataLocation(**(process_scan_files | {field: Path("/nonexisting/file.png")}))
    assert "Path does not point to a file" in str(exc_info.value)


@pytest.mark.parametrize("field", ["x3p_image", "preview_image", "surfacemap_image"])
def test_process_scan_different_directories_raises_error(
    field: str, tmp_path: Path, process_scan_files: dict[str, Path]
) -> None:
    """Test that ProcessedDataLocation raises error when all files are in different directories."""
    # Arrange
    file = tmp_path / "file.png"
    file.touch()
    assert file.parent != process_scan_files[field].parent

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ProcessedDataLocation(**(process_scan_files | {field: file}))
    assert "All fields must point to the same output directory" in str(exc_info.value)
