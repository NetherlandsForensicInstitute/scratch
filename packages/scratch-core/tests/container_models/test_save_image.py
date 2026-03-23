from pathlib import Path

import pytest
from pydantic import ValidationError

from container_models.models import NormalizationBounds
from container_models.scan_image import ScanImage


def test_save_image_to_temp_dir(scan_image: ScanImage, tmp_path: Path) -> None:
    # Arrange
    file_path = tmp_path / "some_image.png"
    scaling = NormalizationBounds(low=0, high=255)
    # Act
    scan_image.save_as_image(output_path=file_path, normalization_bounds=scaling)
    # Arrange
    assert file_path.exists()
    assert file_path.is_file()


@pytest.mark.parametrize(
    ("scale_min", "scale_max"),
    (
        pytest.param(-1, 200, id="scale_min needs to be above 0"),
        pytest.param(0, 300, id="scale_max should be lower then 255"),
        pytest.param(50, 20, id="max value should be higher then the lower value"),
    ),
)
def test_save_image_to_temp_dir_with_wrong_scaling_raises_error(
    scale_min: int, scale_max: int, scan_image: ScanImage, tmp_path: Path
):
    # Arrange
    file_path = tmp_path / "some_image.png"
    # Act
    with pytest.raises(ValidationError):
        scan_image.save_as_image(
            output_path=file_path,
            normalization_bounds=NormalizationBounds(low=scale_min, high=scale_max),
        )
