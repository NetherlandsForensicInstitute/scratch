from pathlib import Path, PosixPath

import numpy as np
import pytest

from image_generation.data_formats import ScanImage
from parsers import load_scan_image, save_to_x3p
from utils.array_definitions import ScanMap2DArray

from ..constants import PRECISION, SCANS_DIR  # type: ignore


def validate_image(
    parsed_image: ScanImage, expected_image_data: ScanMap2DArray, expected_scale: float
):
    """Validate a parsed image."""
    assert isinstance(parsed_image, ScanImage)
    assert parsed_image.data.shape == expected_image_data.shape
    assert np.allclose(
        parsed_image.data,
        expected_image_data,
        equal_nan=True,
        atol=PRECISION,
    )
    assert parsed_image.data.dtype == np.float64
    assert np.isclose(parsed_image.scale_x, parsed_image.scale_y, atol=PRECISION)
    assert np.isclose(parsed_image.scale_x, expected_scale, atol=PRECISION)


@pytest.mark.parametrize(
    "filename, expected_scale",
    [("circle.al3d", 8.7654321e-7), ("circle.x3p", 8.7654321e-7)],
)
def test_parser_can_parse(
    filename: Path, scan_image: ScanMap2DArray, expected_scale: float
):
    file_to_test = SCANS_DIR / filename
    parsed_image = load_scan_image(file_to_test)
    validate_image(
        parsed_image=parsed_image,
        expected_image_data=scan_image,
        expected_scale=expected_scale,
    )


def test_parsed_image_can_be_exported_to_x3p(
    scan_map_2d: ScanImage, tmp_path: PosixPath
):
    save_to_x3p(image=scan_map_2d, output_path=tmp_path / "export.x3p")

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "export.x3p"


@pytest.mark.integration
def test_al3d_can_be_converted_to_x3p(tmp_path: PosixPath):
    al3d_file = SCANS_DIR / "circle.al3d"
    parsed_image = load_scan_image(al3d_file)
    output_file = tmp_path / "export.x3p"
    save_to_x3p(image=parsed_image, output_path=output_file)
    parsed_exported_image = load_scan_image(output_file)
    # compare the parsed data from the exported .x3p file to the parsed data from the .al3d file
    validate_image(
        parsed_image=parsed_exported_image,
        expected_image_data=parsed_image.data,
        expected_scale=parsed_image.scale_x,
    )
