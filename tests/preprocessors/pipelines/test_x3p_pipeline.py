from pathlib import Path

import numpy as np
import pytest
from container_models import ImageContainer
from scipy.constants import femto

from preprocessors.pipelines import parse_scan_pipeline, x3p_pipeline


@pytest.mark.integration
class TestX3pPipeline:
    """Integration tests for x3p_pipeline function."""

    def test_convert_scan_to_x3p_success(self, parsed_al3d_file: ImageContainer, tmp_path: Path) -> None:
        """Test that a ImageContainer is successfully converted to X3P format."""
        # Arrange
        output_path = tmp_path / "output.x3p"

        # Act
        result_path = x3p_pipeline(parsed_al3d_file, output_path)

        # Assert
        assert output_path == result_path
        assert output_path.is_file()
        assert output_path.stat().st_size > 0

    # TODO: can we assert this differently?
    def test_output_file_is_valid_x3p(self, parsed_al3d_file: ImageContainer, tmp_path: Path) -> None:
        """Test that the output file can be parsed back as a valid X3P file."""
        # Arrange
        output_path = tmp_path / "output.x3p"

        # Act
        x3p_pipeline(parsed_al3d_file, output_path)

        # Assert - verify we can parse the generated X3P file
        reparsed_scan = parse_scan_pipeline(output_path, 1, 1)
        scale = reparsed_scan.metadata.scale
        assert isinstance(reparsed_scan, ImageContainer)
        assert reparsed_scan.data.shape == parsed_al3d_file.data.shape
        assert np.isclose(scale.x, parsed_al3d_file.metadata.scale.x, atol=femto)
        assert np.isclose(scale.y, parsed_al3d_file.metadata.scale.y, atol=femto)
        assert np.allclose(reparsed_scan.data, parsed_al3d_file.data, atol=femto, equal_nan=True)
