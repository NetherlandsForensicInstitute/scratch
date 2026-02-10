from pathlib import Path

import numpy as np
import pytest
from container_models.scan_image import ScanImage
from conversion.leveling import SurfaceTerms
from parsers import parse_to_x3p, save_x3p
from scipy.constants import micro
from utils.constants import RegressionOrder

from preprocessors.controller import apply_changes_on_scan_image
from preprocessors.schemas import EditImage


@pytest.fixture
def scan_image():
    width, height = 3, 3
    data = np.ones((height, width), dtype=float)
    data[1, 1] = 6
    scan_image = ScanImage(
        data=data,
        scale_x=1,
        scale_y=1,
    )
    return scan_image


@pytest.fixture
def resample_twice_bigger(scan_image: ScanImage, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    scan_file = tmp_path / "scan.x3p"
    save_x3p(output_path=scan_file, x3p=parse_to_x3p(scan_image).unwrap())

    params = EditImage(
        project_name="test",
        scan_file=scan_file,
        mask=((True, True, True), (True, True, True)),
        cutoff_length=2 * micro,
        resampling_factor=0.5,
        terms=SurfaceTerms.PLANE,
        regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        crop=False,
        step_size_x=1,
        step_size_y=1,
    )

    def assertions(result: ScanImage):
        assert result.data.shape[0] == scan_image.width * 2
        assert "skipping masking, Mask area is not containing any masking fields." in caplog.messages

    return params, assertions


@pytest.fixture
def mask_middle_pixel(scan_image: ScanImage, tmp_path: Path):
    scan_file = tmp_path / "scan.x3p"
    save_x3p(output_path=scan_file, x3p=parse_to_x3p(scan_image).unwrap())

    params = EditImage(
        project_name="test",
        scan_file=scan_file,
        mask=((True, True, True), (True, False, True), (True, True, True)),
        cutoff_length=2 * micro,
        resampling_factor=1,
        terms=SurfaceTerms.PLANE,
        regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        crop=False,
        step_size_x=1,
        step_size_y=1,
    )

    def assertions(result: ScanImage):
        assert np.isnan(result.data[1, 1]), "Pixel value in the middle needs to be masked out."

    return params, assertions


@pytest.fixture
def crop_to_middle_pixel(scan_image: ScanImage, tmp_path: Path):
    scan_file = tmp_path / "scan.x3p"
    save_x3p(output_path=scan_file, x3p=parse_to_x3p(scan_image).unwrap())

    params = EditImage(
        project_name="test",
        scan_file=scan_file,
        mask=((False, False, False), (False, True, False), (False, False, False)),
        cutoff_length=2 * micro,
        resampling_factor=1,
        terms=SurfaceTerms.PLANE,
        regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        crop=True,
        step_size_x=1,
        step_size_y=1,
    )

    def assertions(result: ScanImage):
        assert result.data.shape == (1, 1), "cropped to the middle pixel"

    return params, assertions


@pytest.fixture
def crop_to_resized_image(scan_image: ScanImage, tmp_path: Path):
    scan_file = tmp_path / "scan.x3p"
    save_x3p(output_path=scan_file, x3p=parse_to_x3p(scan_image).unwrap())

    params = EditImage(
        project_name="test",
        scan_file=scan_file,
        mask=((False, False, False), (False, True, False), (False, False, False)),
        cutoff_length=2 * micro,
        resampling_factor=0.5,
        terms=SurfaceTerms.PLANE,
        regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        crop=True,
        step_size_x=1,
        step_size_y=1,
    )

    def assertions(result: ScanImage):
        assert result.data.shape == (2, 2), "cropped to the middle pixel (1,1) but double the size (2,2)"

    return params, assertions


@pytest.mark.parametrize(
    "fixture_name", ["resample_twice_bigger", "mask_middle_pixel", "crop_to_middle_pixel", "crop_to_resized_image"]
)
def test_apply_change_on_scan_image(fixture_name: str, request: pytest.FixtureRequest, scan_image: ScanImage) -> None:
    """Test the different parameters of EditScan in apply_changes_on_scan_image."""
    # Arrange
    params, assertions = request.getfixturevalue(fixture_name)

    # Act
    result = apply_changes_on_scan_image(scan_image=scan_image, edit_image_params=params)
    # Assert
    assertions(result)
