from pathlib import Path

import numpy as np
import pytest
from container_models import ImageContainer

from preprocessors.pipelines import parse_scan_pipeline


@pytest.mark.integration
class TestParseScanPipeline:
    @pytest.mark.parametrize(
        "extension",
        [".al3d", ".x3p"],
    )
    def test_parse_supported_file_success(self, extension: str, scan_directory: Path) -> None:
        """Test that supported file formats are parsed successfully."""
        # Act
        result = parse_scan_pipeline((scan_directory / "circle").with_suffix(extension), 1, 1)

        # Assert
        assert isinstance(result, ImageContainer)
        assert result.data.size > 0
        height, width = result.data.shape
        assert height > 0
        assert width > 0
        assert np.isfinite(result.scale_x)
        assert np.isfinite(result.scale_y)
