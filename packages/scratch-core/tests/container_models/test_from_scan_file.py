from math import ceil
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import micro
from surfalize import Surface
from unittest.mock import patch
from container_models.image import ImageContainer


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
        ImageContainer.from_scan_file.cache_clear()

    def test_load_scan_data_matches_size(self, filepath: Path) -> None:
        # Arrange
        surface = Surface.load(filepath)
        # Act
        scan_image = ImageContainer.from_scan_file(filepath)

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
        data = np.zeros((2, 2))
        step_x = step_y = 1

    class FakeSurfaceTwo:
        data = np.zeros((3, 3))
        step_x = step_y = 1

    @pytest.fixture(autouse=True)
    def empty_cache_for_test(self):
        ImageContainer.from_scan_file.cache_clear()
        yield

    def test_load_scan_image_is_cached(self, tmp_path: Path) -> None:
        # Arrange
        scan_file = tmp_path / "scan.x3p"
        with patch(
            "container_models.image.Surface.load", return_value=self.FakeSurfaceOne()
        ) as mock_load:
            # Act
            image_1 = ImageContainer.from_scan_file(scan_file)
            image_2 = ImageContainer.from_scan_file(scan_file)

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
            "container_models.image.Surface.load",
            side_effect=[
                self.FakeSurfaceOne(),
                self.FakeSurfaceTwo(),
            ],
        ):
            # Act
            ImageContainer.from_scan_file(scan_file_1)
            ImageContainer.from_scan_file(scan_file_1)
            ImageContainer.from_scan_file(scan_file_2)

        # Assert
        info = ImageContainer.from_scan_file.cache_info()
        assert info.hits == 1, "one cache hit expected"
        assert info.misses == 2, "two different files loaded"
        assert info.currsize == 1, "Cache should only hold one item"
