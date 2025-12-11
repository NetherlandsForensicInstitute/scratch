from pathlib import Path

import numpy as np
import pytest
from container_models.scan_image import ScanImage
from pydantic import ValidationError

from preprocessors.pipelines import parse_scan_pipeline
from preprocessors.schemas import UploudScanParameters


@pytest.mark.integration
class TestParseScanPipeline:
    @pytest.mark.parametrize(
        "extension",
        [".al3d", ".x3p"],
    )
    def test_parse_supported_file_success(
        self, extension: str, scan_directory: Path, default_parameters: UploudScanParameters
    ) -> None:
        """Test that supported file formats are parsed successfully."""
        # Act
        result = parse_scan_pipeline((scan_directory / "circle").with_suffix(extension), default_parameters)

        # Assert
        assert isinstance(result, ScanImage)
        assert result.data.size > 0
        height, width = result.data.shape
        assert height > 0
        assert width > 0
        assert np.isfinite(result.scale_x)
        assert np.isfinite(result.scale_y)

    def test_parse_result_is_immutable(self, scan_directory: Path, default_parameters: UploudScanParameters) -> None:
        """Test that ScanImage is immutable and cannot be modified after creation."""
        # Act
        result = parse_scan_pipeline(scan_directory / "circle.x3p", default_parameters)

        # Assert
        # ScanImage should be frozen, so attempting to modify should fail
        with pytest.raises(ValidationError, match="Instance is frozen"):
            result.scale_x = 999.0
