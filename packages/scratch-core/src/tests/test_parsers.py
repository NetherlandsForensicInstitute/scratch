from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
from parsers import parse_file


def test_file_can_be_parsed(scans_dir: Path, image_data: NDArray):
    def validate_image(path, parsed_image) -> bool:
        return (
            parsed_image.data == pytest.approx(image_data)
            and parsed_image.data.dtype == np.float64
            and 0 < parsed_image.scale_x == parsed_image.scale_y
            and parsed_image.path_to_original_image == path
        )

    assert all(
        validate_image(path, parse_file(path))
        for path in scans_dir.iterdir()
        if path.is_file()
    )
