from dataclasses import dataclass
from math import ceil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.constants import micro
from surfalize import Surface

from container_models.scan_image import ScanImage
from parsers.loaders import _load_surface


@pytest.fixture(scope="class")
def filepath(scans_dir: Path, request: pytest.FixtureRequest):
    return scans_dir / request.param


@pytest.mark.parametrize(
    "filepath",
    [
        "Klein_non_replica_mode.al3d",
        "Klein_non_replica_mode_X3P_Scratch.x3p",
    ],
    indirect=True,
)
class TestLoadScanImage:
    def test_load_scan_data_matches_size(self, filepath: Path) -> None:
        # Arrange
        surface = Surface.load(filepath)
        # Act
        result = ScanImage.from_file(filepath)
        scan_image = result
        # Assert
        assert scan_image.data.shape == (
            ceil(surface.data.shape[0]),
            ceil(surface.data.shape[1]),
        )
        assert scan_image.scale_y == surface.step_y * micro
        assert scan_image.scale_x == surface.step_x * micro


class TestLoadScanImageCaching:
    @dataclass
    class FakeSurfaceOne:
        data = np.zeros((2, 2))
        step_x = 1
        step_y = 1
        metadata = {"some_data": "data"}

    class FakeSurfaceTwo:
        data = np.zeros((2, 2))
        step_x = 1
        step_y = 1
        metadata = {"some_data": "data"}

    @pytest.fixture(autouse=True)
    def empty_cache_for_test(self):
        _load_surface.cache_clear()
        yield

    def test_load_scan_image_is_cached(self, tmp_path: Path) -> None:
        # Arrange
        scan_file = tmp_path / "scan.x3p"
        with patch(
            "parsers.loaders.Surface.load", return_value=self.FakeSurfaceOne()
        ) as mock_load:
            # Act
            image_1 = ScanImage.from_file(scan_file)
            image_2 = ScanImage.from_file(scan_file)

        # Assert
        assert image_1 is not image_2, (
            "Same data should be cached, but new object should be created."
        )
        assert mock_load.call_count == 1, (
            "Surface.load should be called only once due to caching"
        )

    def test_load_scan_image_only_caches_one_image(self, tmp_path: Path) -> None:
        # Arrange
        scan_file_1 = tmp_path / "scan_1.x3p"
        scan_file_2 = tmp_path / "scan_2.x3p"

        with patch(
            "parsers.loaders.Surface.load",
            side_effect=[
                self.FakeSurfaceOne(),
                self.FakeSurfaceTwo(),
            ],
        ):
            # Act
            _image_1 = ScanImage.from_file(scan_file_1)
            _image_2 = ScanImage.from_file(scan_file_1)
            _image_3 = ScanImage.from_file(scan_file_2)

        # Assert
        info = _load_surface.cache_info()
        assert info.hits == 1, "one cache hit expected"
        assert info.misses == 2, "two different files loaded"
        assert info.currsize == 1, "Cache should only hold one item"
