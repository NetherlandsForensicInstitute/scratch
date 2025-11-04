from pathlib import Path, PosixPath

import numpy as np
from numpy.typing import NDArray
import pytest
from parsers import (
    parse_surface_scan_file,
    ParsedImage,
    parse_image,
    parse_scan,
    save_to_x3p,
)


def validate_image(
    path_to_original_image: Path, parsed_image: ParsedImage, image_data: NDArray
) -> bool:
    """Validate a parsed image."""
    return (
        isinstance(parsed_image, ParsedImage)
        and parsed_image.data == pytest.approx(image_data)
        and parsed_image.data.dtype == np.float64
        and 0 < parsed_image.scale_x == parsed_image.scale_y
        and parsed_image.path_to_original_image == path_to_original_image
    )


def test_image_parser_can_parse_png(png_file: Path, image_data: NDArray):
    parsed_image = parse_image(png_file)
    assert validate_image(
        path_to_original_image=png_file,
        parsed_image=parsed_image,
        image_data=image_data,
    )


def test_scan_parser_can_parse_x3p(x3p_file: Path, image_data: NDArray):
    parsed_image = parse_scan(x3p_file)
    assert validate_image(
        path_to_original_image=x3p_file,
        parsed_image=parsed_image,
        image_data=image_data,
    )


def test_scan_parser_can_parse_al3d(al3d_file: Path, image_data: NDArray):
    parsed_image = parse_scan(al3d_file)
    assert validate_image(
        path_to_original_image=al3d_file,
        parsed_image=parsed_image,
        image_data=image_data,
    )


def test_x3p_file_can_be_exported(image_data: NDArray, tmp_path: PosixPath):
    parsed_image = ParsedImage(
        data=image_data, path_to_original_image=Path("some/path/file.x3p")
    )
    output_file = tmp_path / "export.x3p"
    save_to_x3p(image=parsed_image, path=output_file)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "export.x3p"


@pytest.mark.integration
def test_scan_files_can_be_parsed(scans_dir: Path, image_data: NDArray):
    assert all(
        validate_image(
            path_to_original_image=path,
            parsed_image=parse_surface_scan_file(path),
            image_data=image_data,
        )
        for path in scans_dir.iterdir()
        if path.is_file()
    )


@pytest.mark.integration
def test_exported_x3p_file_can_be_parsed(al3d_file: Path, tmp_path: PosixPath):
    parsed_image = parse_surface_scan_file(al3d_file)
    output_file = tmp_path / "export.x3p"
    save_to_x3p(image=parsed_image, path=output_file)
    parsed_exported_image = parse_surface_scan_file(output_file)
    assert validate_image(
        path_to_original_image=output_file,
        parsed_image=parsed_exported_image,
        image_data=parsed_image.data,
    )
