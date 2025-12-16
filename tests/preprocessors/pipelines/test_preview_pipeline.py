from pathlib import Path

import pytest
from container_models.scan_image import ScanImage
from PIL import Image

from preprocessors.pipelines import preview_pipeline


@pytest.mark.integration
class TestPreviewPipeline:
    """Integration tests for preview_pipeline function."""

    def test_generate_preview_success(self, parsed_al3d_file: ScanImage, tmp_path: Path) -> None:
        """Test that a preview image is successfully generated from scan data."""
        # Arrange
        output_path = tmp_path / "preview.png"

        # Act
        result_path = preview_pipeline(parsed_al3d_file, output_path)

        # Assert
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.is_file()
        assert output_path.stat().st_size > 0

    def test_preview_is_valid_png_image(self, parsed_al3d_file: ScanImage, tmp_path: Path) -> None:
        """Test that the generated preview file is a valid PNG image that can be opened."""
        # Act
        preview = preview_pipeline(parsed_al3d_file, output_path=tmp_path / "preview.png")

        # Assert - verify we can open the PNG file
        with Image.open(preview) as img:
            assert img.format == "PNG"
            assert img.size == parsed_al3d_file.data.shape
