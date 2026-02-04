from math import ceil
from pathlib import Path

import pytest
from scipy.constants import micro
from surfalize import Surface
from unittest.mock import patch

from parsers import load_scan_image

from ..helper_function import unwrap_result


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
class TestLoadImageContainer:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the load_scan_image cache before each test."""
        load_scan_image.cache_clear()
        yield

    def test_load_scan_data_matches_size(self, filepath: Path) -> None:
        # Arrange
        surface = Surface.load(filepath)
        # Act
        result = load_scan_image(filepath)
        scan_image = unwrap_result(result)

        # Assert
        assert scan_image.data.shape == (
            ceil(surface.data.shape[0]),
            ceil(surface.data.shape[1]),
        )
        assert scan_image.metadata.scale == (
            surface.step_x * micro,
            surface.step_y * micro,
        )


class TestLoadImageContainerCaching:
    class FakeSurfaceOne:
        pass

    class FakeSurfaceTwo:
        pass

    @pytest.fixture(autouse=True)
    def empty_cache_for_test(self):
        load_scan_image.cache_clear()
        yield

    def test_load_scan_image_is_cached(self, tmp_path: Path) -> None:
        # Arrange
        scan_file = tmp_path / "scan.x3p"
        with patch(
            "parsers.loaders.Surface.load", return_value=self.FakeSurfaceOne()
        ) as mock_load:
            # Act
            image_1 = load_scan_image(scan_file)
            image_2 = load_scan_image(scan_file)

        # Assert
        assert image_1 is image_2, "same object expected due to caching"
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
            load_scan_image(scan_file_1)
            load_scan_image(scan_file_1)
            load_scan_image(scan_file_2)

        # Assert
        info = load_scan_image.cache_info()
        assert info.hits == 1, "one cache hit expected"
        assert info.misses == 2, "two different files loaded"
        assert info.currsize == 1, "Cache should only hold one item"
