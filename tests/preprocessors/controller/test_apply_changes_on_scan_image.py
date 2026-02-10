from pathlib import Path

import numpy as np
from container_models.scan_image import ScanImage
from conversion.leveling import SurfaceTerms
from parsers import parse_to_x3p, save_x3p
from utils.constants import RegressionOrder

from preprocessors.controller import apply_changes_on_scan_image
from preprocessors.schemas import EditImage


def test_apply_change_on_scan_image(tmp_path: Path) -> None:
    """A function to test all."""
    # Arrange
    data = np.ones((3, 3), dtype=float)
    data[1, 1] = 6
    scan_image = ScanImage(
        data=data,
        scale_x=1,
        scale_y=1,
    )
    scan_file = tmp_path / "scan.x3p"
    save_x3p(output_path=scan_file, x3p=parse_to_x3p(scan_image).unwrap())
    params = EditImage(
        project_name="test",
        scan_file=tmp_path / "scan.x3p",
        mask=((True, True, True), (True, True, True)),
        cutoff_length=2 * 1e-6,
        resampling_factor=0.5,
        terms=SurfaceTerms.PLANE,
        regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        crop=False,
        step_size_x=1,
        step_size_y=1,
    )

    # Act
    result = apply_changes_on_scan_image(
        scan_image=scan_image, edit_image_params=params, mask=np.zeros(scan_image.data.shape, dtype=np.bool_)
    )
    # Assert
    print(result)
