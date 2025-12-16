from pathlib import Path

import numpy as np
import pytest
from fastapi.exceptions import HTTPException
from image_generation.data_formats import ScanImage

from preprocessors.helpers import export_image_pipeline


def test_export_image_pipeline_raises_error():
    # Arrange
    def faker(*args, **kwargs) -> None:
        raise ValueError("Boem!")

    expected_http_code = 500
    # Act
    with pytest.raises(HTTPException) as exc_info:
        export_image_pipeline(Path(), faker, np.array([]))  # pyright: ignore
    # Assert
    assert exc_info.value.status_code == expected_http_code
    assert exc_info.value.detail == "Failed to generate : Boem!"


def test_export_image_pipeline_creates_file(tmp_path: Path):
    # Arrange
    def faker(*args, **kwargs) -> ScanImage:
        return ScanImage(data=np.array([[1.0, 2.0]]), scale_x=1, scale_y=1)

    # Act
    export_image_pipeline(tmp_path / "test.png", faker, np.array([]))  # pyright: ignore
    # Assert
    assert (tmp_path / "test.png").exists()
