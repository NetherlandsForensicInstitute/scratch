from pathlib import Path, PosixPath

import numpy as np
from numpy.typing import NDArray
import pytest
from parsers import ScanImage, save_to_x3p


def validate_image(
    path_to_original_image: Path,
    parsed_image: ScanImage,
    expected_image_data: NDArray,
    expected_scale: float,
) -> bool:
    """Validate a parsed image."""
    return (
        isinstance(parsed_image, ScanImage)
        and parsed_image.data == pytest.approx(expected_image_data)
        and parsed_image.data.dtype == np.float64
        and parsed_image.path_to_original_image == path_to_original_image
        and parsed_image.scale_x == pytest.approx(parsed_image.scale_y)
        and parsed_image.scale_x == pytest.approx(expected_scale)
    )


def test_exception_on_incorrect_file_extension():
    with pytest.raises(ValueError, match="extension"):
        _ = ScanImage.from_file(Path("export.txt"))


def test_exception_on_incorrect_shape(image_data: NDArray):
    with pytest.raises(ValueError, match="shape"):
        _ = ScanImage(
            data=np.expand_dims(image_data, -1),
            path_to_original_image=Path("export.x3p"),
        )


def test_parser_can_parse_png(png_file: Path, image_data: NDArray):
    parsed_image = ScanImage.from_file(png_file)
    assert validate_image(
        path_to_original_image=png_file,
        parsed_image=parsed_image,
        expected_image_data=image_data,
        expected_scale=1.0,
    )


def test_parser_can_parse_x3p(x3p_file: Path, image_data: NDArray):
    parsed_image = ScanImage.from_file(x3p_file)
    assert validate_image(
        path_to_original_image=x3p_file,
        parsed_image=parsed_image,
        expected_image_data=image_data,
        expected_scale=0.87654321,
    )


def test_parser_can_parse_al3d(al3d_file: Path, image_data: NDArray):
    parsed_image = ScanImage.from_file(al3d_file)
    assert validate_image(
        path_to_original_image=al3d_file,
        parsed_image=parsed_image,
        expected_image_data=image_data,
        expected_scale=0.87654321,
    )


def test_parsed_image_can_be_exported_to_x3p(
    scan_image: ScanImage, tmp_path: PosixPath
):
    save_to_x3p(image=scan_image, output_path=tmp_path / "export.x3p")
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "export.x3p"


@pytest.mark.integration
def test_al3d_can_be_converted_to_x3p(al3d_file: Path, tmp_path: PosixPath):
    parsed_image = ScanImage.from_file(al3d_file)
    output_file = tmp_path / "export.x3p"
    save_to_x3p(image=parsed_image, output_path=output_file)
    parsed_exported_image = ScanImage.from_file(output_file)
    # compare the parsed data from the exported .x3p file to the parsed data from the .al3d file
    assert validate_image(
        path_to_original_image=output_file,
        parsed_image=parsed_exported_image,
        expected_image_data=parsed_image.data,
        expected_scale=parsed_image.scale_x,
    )
