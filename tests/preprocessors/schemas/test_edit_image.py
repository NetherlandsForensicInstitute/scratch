from pathlib import Path

import pytest
from pydantic import ValidationError

from preprocessors.schemas import EditImage, EditImageParameters


class TestEditImage:
    """Tests for EditImage request model."""

    def test_should_create_with_valid_x3p_file(
        self, scan_directory: Path, edit_image_parameters: EditImageParameters
    ) -> None:
        """Test that EditImage can be created with a valid X3P file."""
        # Arrange
        x3p_file = scan_directory / "circle.x3p"

        # Act
        edit_image = EditImage(  # type: ignore
            scan_file=x3p_file,
            parameters=edit_image_parameters,
        )

        # Assert
        assert edit_image.scan_file == x3p_file
        assert edit_image.project_name is None
        assert edit_image.tag == edit_image.scan_file.stem

    def test_should_create_with_project_name(
        self, scan_directory: Path, edit_image_parameters: EditImageParameters
    ) -> None:
        """Test that EditImage can be created with optional project_name."""
        # Arrange
        x3p_file = scan_directory / "circle.x3p"
        project_name = "forensic_analysis_2026"

        # Act
        edit_image = EditImage(scan_file=x3p_file, project_name=project_name, parameters=edit_image_parameters)

        # Assert
        assert edit_image.project_name == edit_image.tag == project_name

    def test_should_reject_non_x3p_file(self, scan_directory: Path, edit_image_parameters: EditImageParameters) -> None:
        """Test that EditImage rejects files that are not X3P format."""
        # Arrange
        al3d_file = scan_directory / "circle.al3d"

        # Act & Assert
        with pytest.raises(ValidationError, match="unsupported extension") as exc_info:
            EditImage(
                scan_file=al3d_file,
                parameters=edit_image_parameters,  # type: ignore
            )

        # Assert
        errors = exc_info.value.errors()
        error_messages = [str(error["ctx"]["error"]) for error in errors if "ctx" in error]
        assert any("unsupported extension" in msg for msg in error_messages)

    def test_should_reject_non_existent_file(self, tmp_path: Path, edit_image_parameters: EditImageParameters) -> None:
        """Test that EditImage rejects non-existent files."""
        # Arrange
        non_existent_file = tmp_path / "does_not_exist.x3p"

        # Act & Assert
        with pytest.raises(ValidationError, match="Path does not point to a file") as exc_info:
            EditImage(
                scan_file=non_existent_file,
                parameters=edit_image_parameters,  # type: ignore
            )

        # Assert
        errors = exc_info.value.errors()
        assert any("scan_file" in error["loc"] for error in errors)
