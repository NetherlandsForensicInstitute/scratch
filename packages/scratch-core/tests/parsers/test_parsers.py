from pathlib import Path

import numpy as np
import pytest

from image_generation.data_formats import ScanImage
from parsers import from_file, save_x3p, parse_to_x3p
from utils.array_definitions import ScanMap2DArray

from ..constants import PRECISION, SCANS_DIR


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
    filename: Path, image_data: ScanMap2DArray, expected_scale: float
):
    _ = from_file(SCANS_DIR / filename).bind(
        lambda image: validate_image(
            parsed_image=image,
            expected_image_data=image_data,
            expected_scale=expected_scale,
        )  # type: ignore
    )


@pytest.mark.integration
def test_al3d_can_be_converted_to_x3p(tmp_path: Path):
    al3d_file = SCANS_DIR / "circle.al3d"
    output_file = tmp_path / "export.x3p"
    _ = (
        from_file(al3d_file)
        .bind(lambda array2d: parse_to_x3p(array2d))
        .bind(
            lambda x3p: save_x3p(x3p, output_file)  # type: ignore
        )
    )

    # Verify the file was actually created (IO side effect occurred)
    assert output_file.exists()
