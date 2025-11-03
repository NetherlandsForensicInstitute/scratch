import os
from pathlib import Path

import numpy as np
import pytest

from parsers import parse_file
from tests.conftest import SCANS_DIR


@pytest.mark.parametrize("path", [SCANS_DIR / f for f in os.listdir(SCANS_DIR)])
def test_file_can_be_parsed(path: Path, image_data: np.ndarray):
    parsed_image = parse_file(path)
    assert parsed_image.data == pytest.approx(image_data)
    assert parsed_image.data.dtype == np.float64
    assert 0 < parsed_image.scale_x == parsed_image.scale_y
    assert parsed_image.path_to_original_image == path
